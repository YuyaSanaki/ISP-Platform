### English

`core/rank_edit.py`: OE moves genes to the front and trims the tail (length-preserving; a later step sits left of earlier ones), KD removes genes (shorter). Used only by State-feedback ISP; conventional ISP has its own edits in the perturber.

### 日本語

`core/rank_edit.py`: OE は遺伝子を先頭へ移して末尾を切り（長さを保つ。後のステップほど左に来る）、KD は遺伝子を除きます（短くなる）。State-feedback ISP だけが使い、従来型 ISP は perturber 内の独自の編集を使います。
