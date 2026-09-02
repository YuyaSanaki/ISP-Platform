"""
Geneformer tokenizer.

Input data:
Required format: raw counts scRNAseq data without feature selection as .loom file
Required row (gene) attribute: "ensembl_id"; Ensembl ID for each gene
Required col (cell) attribute: "n_counts"; total read counts in that cell
Optional col (cell) attribute: "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria
Optional col (cell) attributes: any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below

Usage:
  from geneformer import TranscriptomeTokenizer
  tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("loom_data_directory", "output_directory", "output_prefix")
"""

from __future__ import annotations
from typing import Literal
import pickle
from pathlib import Path

import logging

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import anndata as ad
import loompy as lp
import numpy as np
import scipy.sparse as sp
from datasets import Dataset

from time import time

import csv
import sys

logger = logging.getLogger(__name__)


# setting 
USE_GPU = "cuda:0"

# Dictionary files — prefer core/geneformer/dicts/mouse/, fall back to legacy flat dicts/.
_DICTS_DIR = Path(__file__).parent / "dicts"


def _resolve_dict_path(*candidates: str) -> Path:
    for name in candidates:
        for base in (_DICTS_DIR / "mouse", _DICTS_DIR):
            path = base / name
            if path.is_file():
                return path
    return _DICTS_DIR / "mouse" / candidates[0]


GENE_MEDIAN_FILE = _resolve_dict_path("mouse_gene_median_dictionary.pkl")
TOKEN_DICTIONARY_FILE = _resolve_dict_path("MLM-re_token_dictionary_v1.pkl")


def rank_genes(gene_vector, gene_tokens, max_input_size: int = 2048):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)[:max_input_size]

    return gene_tokens[sorted_indices]

def tokenize_cell(gene_vector, gene_tokens, max_input_size: int = 2048):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]

    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask], max_input_size)

def load_not_use_files(csv_file_path) :

    not_use_file_paths = []
    with open(csv_file_path, mode="r") as f :
        reader = csv.reader(f)
        for row in reader :
            not_use_file_paths.append(row[0])
    
    return not_use_file_paths



class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=1,
        max_cells=300_000,
        gene_median_file=None,
        token_dictionary_file=None,
        species_config=None,
        max_input_size=None,
        chunk_size=512,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset.
            Keys are the names of the attributes in the loom file.
            Values are the names of the attributes in the dataset.
        nproc : int
            Number of processes to use for dataset mapping.
        max_cells : int
            Max number of cells to tokenize before saving to disk.
        gene_median_file : Path, optional
            Override path to gene median dictionary pickle.
        token_dictionary_file : Path, optional
            Override path to token dictionary pickle.
        species_config : dict, optional
            ``species`` block from pipeline YAML (model_organism, model, human_variant).
            Resolves backend dict paths and enables ortholog conversion when needed.
        max_input_size : int, optional
            Max tokens per cell (2048 mouse, 4096 human V2). Inferred from species when omitted.
        chunk_size : int
            Cells per loom scan / ortholog-conversion batch (avoids full densify).
        """
        from geneformer.backends import get_backend, parse_species_config
        from geneformer.species_context import should_convert

        self.species_config = parse_species_config(species_config) if species_config else None
        self._species_convert = should_convert(self.species_config)

        if self.species_config:
            backend = get_backend(self.species_config)
            gene_median_file = gene_median_file or backend.gene_median_dictionary
            token_dictionary_file = token_dictionary_file or backend.token_dictionary
            max_input_size = max_input_size or backend.max_input_size

        gene_median_file = gene_median_file or GENE_MEDIAN_FILE
        token_dictionary_file = token_dictionary_file or TOKEN_DICTIONARY_FILE
        self.max_input_size = int(max_input_size or 2048)
        self.chunk_size = max(1, int(chunk_size))
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        self._refresh_vocabulary_keys()

        self.gene_median_file_path_dict = {}
        self.start_reading_file_num = 0

        # total cell nums
        self.learning_cell_nums = 22_446_161

        # final loom file flag
        self.last_dataset_flag = False

        # max cells in 1 dataset
        self.max_cells = max_cells

    def _refresh_vocabulary_keys(self):
        """Keep only genes present in both median and token dictionaries."""
        median_keys = set(self.gene_median_dict.keys())
        token_keys = set(self.gene_token_dict.keys())
        vocab_keys = median_keys & token_keys
        missing_token = median_keys - token_keys
        missing_median = token_keys - median_keys
        if missing_token:
            logger.info(
                "Tokenizer: excluding %d gene(s) present in median dict but not token dict.",
                len(missing_token),
            )
        if missing_median:
            logger.debug(
                "Tokenizer: %d token-dict gene(s) have no median entry.",
                len(missing_median),
            )
        self.gene_keys = list(vocab_keys)
        self.genelist_dict = dict.fromkeys(self.gene_keys, True)

    def tokenize_data(
        self,
        data_directory: Path | str,
        output_directory: Path | str,
        output_prefix: str,
        file_format: Literal["loom", "h5ad"] = "loom",
        use_generator: bool = False,
    ):
        """
        Tokenize .loom files in loom_data_directory and save as tokenized .dataset in output_directory.

        Parameters
        ----------
        loom_data_directory : Path
            Path to directory containing loom files or anndata files
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        file_format : str
            Format of input files. Can be "loom" or "h5ad".
        use_generator : bool
            Whether to use generator or dict for tokenization.
        """

        data_set_num = self.start_reading_file_num
        while(1) :
            
            tokenized_cells, cell_metadata = self.tokenize_files(
                data_set_num, data_directory, file_format
            )
            if int(len(tokenized_cells)) == 0 :
                continue
            
            tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator)
            
            output_path = output_directory+"/"+output_prefix+"_"+str(data_set_num)+".dataset"
            tokenized_dataset.save_to_disk(output_path)
            print("saved to {}".format(output_path))
            
            if self.last_dataset_flag == True :
                break

            data_set_num += 1
        


    def tokenize_files(
        self, data_set_num, data_directory, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )

        # Path(str) drops a trailing slash; always resolve via Path.glob so the
        # file count and the iteration target the same directory listing.
        # Keep filesystem (unsorted) order to match upstream/WebUI — sorted() would
        # change cell order and thus max_cells / ISP max_ncells subsets.
        data_dir = Path(data_directory)
        input_files = list(data_dir.glob(f"*.{file_format}"))
        total_loom_datas = len(input_files)
        total_cells_num = 0
        for enum1, file_path in enumerate(input_files):
            if (enum1 < self.start_reading_file_num) : 
                continue
            file_found = 1
            print("=================================")
            print("[{} / {}]".format(enum1, total_loom_datas))
            print("Tokenizing : {}".format(file_path))
            
            file_tokenized_cells, file_cell_metadata, cells_num = tokenize_file_fn(file_path)
            tokenized_cells += file_tokenized_cells
            
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

            total_cells_num += cells_num


            if enum1 == total_loom_datas -1 :
                self.last_dataset_flag = True
            else :
                if total_cells_num >= self.max_cells :
                    self.start_reading_file_num = enum1 + 1
                    break
                else :
                    pass
        

        if file_found == 0:
            msg = f"No .{file_format} files found in directory {data_directory}."
            logger.error(msg)
            raise RuntimeError(msg)

        return tokenized_cells, cell_metadata

    @staticmethod
    def _require_ensembl_row_attr(data, loom_file_path) -> str:
        """Return the loom row-attr key holding Ensembl IDs; never invent one."""
        if "ensembl_id" in data.ra.keys():
            return "ensembl_id"
        raise KeyError(
            f"{loom_file_path} is missing required row attribute 'ensembl_id' "
            f"(found: {list(data.ra.keys())}). Refusing to fall back to "
            "unrelated row attributes."
        )

    def _loom_filter_pass_indices(self, data, loom_file_path) -> np.ndarray:
        try:
            data.ca["filter_pass"]
            var_exists = True
        except AttributeError:
            var_exists = False
        if var_exists:
            return np.where([i == 1 for i in data.ca["filter_pass"]])[0]
        print(
            f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
        )
        return np.arange(data.shape[1], dtype=int)

    def _loom_view_to_anndata(self, view, row_attr_key):
        """
        Build AnnData for one loompy.scan batch (genes × batch_cells → cells × genes).

        Only the current scan batch is densified — never the full loom matrix.
        """
        mat = np.asarray(view[:, :]).T
        obs = {k: np.asarray(view.ca[k]) for k in view.ca.keys()}
        var = {"ensembl_id": np.asarray(view.ra[row_attr_key]).astype(str)}
        adata = ad.AnnData(X=sp.csr_matrix(mat), obs=obs, var=var)
        if "n_counts" not in adata.obs:
            adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        return adata

    def _maybe_convert_adata(self, adata):
        if not self._species_convert:
            return adata
        from geneformer.species_context import remap_adata_ensembl_ids

        adata, _ = remap_adata_ensembl_ids(adata, self.species_config)
        # Ortholog drop / N→1 collapse (policy-dependent) changes per-cell totals; refresh so
        # metadata matches the matrix actually tokenized.
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        return adata

    def _tokenize_adata_matrix(
        self,
        adata,
        *,
        target_sum=10_000,
        chunk_size=None,
        file_cell_metadata=None,
    ):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if self.custom_attr_name_dict is not None and file_cell_metadata is None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        ensembl_ids = adata.var["ensembl_id"].astype(str)
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in ensembl_ids]
        )[0]
        coding_miRNA_ids = ensembl_ids.iloc[coding_miRNA_loc].to_numpy()
        norm_factor_vector = np.array(
            [self.gene_median_dict[i] for i in coding_miRNA_ids]
        )
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        try:
            _ = adata.obs["filter_pass"]
            var_exists = True
        except KeyError:
            var_exists = False

        if var_exists:
            filter_pass_loc = np.where([i == 1 for i in adata.obs["filter_pass"]])[0]
        else:
            filter_pass_loc = np.array([i for i in range(adata.shape[0])])

        tokenized_cells = []
        for i in range(0, len(filter_pass_loc), chunk_size):
            idx = filter_pass_loc[i : i + chunk_size]
            X_view = adata[idx, coding_miRNA_loc].X
            # Match tokenize_loom: divide by the sum of coding/miRNA genes in this
            # cell, not obs["n_counts"] (often pre-conversion / all-gene library size).
            cell_totals = np.asarray(X_view.sum(axis=1), dtype=np.float64).reshape(-1, 1)
            X_norm = X_view / cell_totals * target_sum / norm_factor_vector
            X_norm = sp.csr_matrix(X_norm)
            tokenized_cells += [
                tokenize_cell(
                    X_norm[j].data,
                    coding_miRNA_tokens[X_norm[j].indices],
                    self.max_input_size,
                )
                for j in range(X_norm.shape[0])
            ]
            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
        return tokenized_cells, file_cell_metadata

    def tokenize_anndata(self, adata_file_path, target_sum=10_000, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        adata = ad.read_h5ad(adata_file_path, backed="r")
        adata = self._maybe_convert_adata(adata)

        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        tokenized_cells, file_cell_metadata = self._tokenize_adata_matrix(
            adata,
            target_sum=target_sum,
            chunk_size=chunk_size,
            file_cell_metadata=file_cell_metadata if self.custom_attr_name_dict else None,
        )
        return tokenized_cells, file_cell_metadata

    def tokenize_loom(self, loom_file_path, target_sum=10_000, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.chunk_size
        if self._species_convert:
            # Cross-species: ortholog remap needs AnnData, but never densify the full
            # genes×cells matrix — scan loom in batches, convert+tokenize each batch.
            if self.custom_attr_name_dict is not None:
                file_cell_metadata = {
                    attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
                }
            else:
                file_cell_metadata = None
            tokenized_cells = []
            n_cells_total = 0
            with lp.connect(str(loom_file_path)) as data:
                row_attr_key = self._require_ensembl_row_attr(data, loom_file_path)
                filter_pass_loc = self._loom_filter_pass_indices(data, loom_file_path)
                if self.max_cells is not None and len(filter_pass_loc) > self.max_cells:
                    filter_pass_loc = filter_pass_loc[: int(self.max_cells)]
                n_cells_total = int(len(filter_pass_loc))
                for (_ix, _selection, view) in data.scan(
                    items=filter_pass_loc,
                    axis=1,
                    batch_size=chunk_size,
                ):
                    adata = self._loom_view_to_anndata(view, row_attr_key)
                    adata = self._maybe_convert_adata(adata)
                    cells, file_cell_metadata = self._tokenize_adata_matrix(
                        adata,
                        target_sum=target_sum,
                        chunk_size=chunk_size,
                        file_cell_metadata=file_cell_metadata,
                    )
                    tokenized_cells.extend(cells)
                    del adata
            return tokenized_cells, file_cell_metadata, n_cells_total

        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys() 
            }
        
        # Use existing gene_median_dict if no file-specific dict is provided
        if str(loom_file_path) in self.gene_median_file_path_dict:
            loom_file_median = self.gene_median_file_path_dict[str(loom_file_path)]
            print(f"読み込んだloom fileのmedian file: {loom_file_median}")
            with open(loom_file_median, "rb") as f:
                self.gene_median_dict = pickle.load(f)

        self._refresh_vocabulary_keys()

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            # Geneformer expects 'ensembl_id' as a row attribute
            row_attr_key = self._require_ensembl_row_attr(data, loom_file_path)
            
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra[row_attr_key]]
            )[0]
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra[row_attr_key][coding_miRNA_loc]
                ]
            )
            coding_miRNA_ids = data.ra[row_attr_key][coding_miRNA_loc]
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )
            print(coding_miRNA_tokens.shape[0])

            filter_pass_loc = self._loom_filter_pass_indices(data, loom_file_path)
            cells_counts = int(data.shape[1])

            # scan through .loom files and tokenize cells (batch_size matches upstream)
            tokenized_cells = []
            for (_ix, _selection, view) in data.scan(
                items=filter_pass_loc,
                axis=1,
                batch_size=chunk_size,
            ):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                    subview[:, :]
                    / np.sum(subview[:, :], axis=0)
                    * target_sum
                    / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                if coding_miRNA_tokens.shape[0] > 0 :
                    tokenized_cells += [
                        tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens, self.max_input_size)
                        for i in range(subview_norm_array.shape[1])
                    ]
                else :
                    pass

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None

        return tokenized_cells, file_cell_metadata, cells_counts

    def create_dataset(self, tokenized_cells, cell_metadata, use_generator=False):
        print("Creating dataset.")
           
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None: # skip
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator:
            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}
            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        max_len = self.max_input_size

        def truncate(example):
            example["input_ids"] = example["input_ids"][:max_len]
            return example
        
        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)
        

        # measure lengths of dataset
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example
        
        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )

        return output_dataset_truncated_w_length