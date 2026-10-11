### English

Stripping `==` pins (both Docker and H100 setup) makes dependency versions non-reproducible; Docker uses Python 3.12 + `datasets==5.0.1`, H100 uses Python 3.11 without that pin.

### 日本語

`==` の固定を外しているため（Docker と H100 の両方）、依存パッケージのバージョンが再現できません。Docker は Python 3.12 + `datasets==5.0.1`、H100 は Python 3.11 でその固定がありません。
