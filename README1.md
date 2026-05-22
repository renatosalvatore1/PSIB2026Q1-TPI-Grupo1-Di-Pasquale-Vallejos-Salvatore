## Dataset

Este proyecto usa el dataset ds003969 de OpenNeuro (EEG meditación).

Para descargarlo:
```bash
pip install datalad
brew install git-annex
datalad install https://github.com/OpenNeuroDatasets/ds003969.git
cd ds003969
datalad get sub-001/eeg/
```