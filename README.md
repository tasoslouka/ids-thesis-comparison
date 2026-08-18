\# Random Forest and Snort DDoS IDS Comparison



This repository contains the experimental code and supporting artefacts for the MSc Applied Cyber Security thesis:



\*\*A Controlled Comparison of Random Forest and Snort for DDoS Intrusion Detection Using CICIDS2017\*\*



The study evaluates two intrusion-detection approaches:



\- Random Forest classification using CICIDS2017 flow-level records.

\- Snort signature-based detection using packet-level CICIDS2017 traffic.



The approaches are evaluated in parallel rather than through a direct row-by-row comparison because their outputs represent different analytical units.



\## Experimental Components



\### Random Forest within-dataset evaluation



The primary machine-learning experiment uses the CICIDS2017 Friday DDoS dataset for binary benign-versus-DDoS classification.



The workflow includes:



\- data cleaning;

\- stratified 80/20 train-test splitting;

\- baseline Random Forest training;

\- hyperparameter optimisation;

\- exact duplicate auditing;

\- deduplicated robustness testing;

\- permutation importance;

\- feature ablation; and

\- misclassification analysis.



\### Snort evaluation



Snort 2.9.20 was evaluated using selected CICIDS2017 DDoS and benign PCAP windows.



The experiments include:



\- attacker-specific rule detection;

\- threshold sensitivity using 50, 100 and 200 matching packets within one second;

\- rule broadening;

\- multiple benign control windows; and

\- analysis of rule-event volume.



Raw CICIDS2017 PCAP files are not included in this repository.



\### Cross-dataset Random Forest transfer



A separate experiment evaluates generic attack-versus-benign transfer between CICIDS2017 and UNSW-NB15.



Ten conceptually corresponding flow features are harmonised before bidirectional evaluation.



This experiment is interpreted as a cross-dataset transfer stress test rather than as DDoS-specific generalisation.



\## Main Scripts



\- `train\_rf\_cicids\_ddos\_tuned.py`

\- `train\_rf\_cicids\_ddos\_deduplicated.py`

\- `audit\_rf\_ddos\_duplicates.py`

\- `rf\_ddos\_permutation\_importance.py`

\- `rf\_ddos\_ablation.py`

\- `rf\_ddos\_misclassification\_analysis.py`

\- `inspect\_ddos\_temporal\_fields.py`

\- `create\_harmonised\_cross\_dataset\_files\_stratified.py`

\- `cross\_dataset\_rf\_harmonised\_log\_scaled.py`



\## Software Environment



Machine-learning experiments:



\- Python 3.10.20

\- NumPy 2.2.6

\- pandas 2.3.3

\- scikit-learn 1.7.2

\- Matplotlib 3.10.9

\- Conda 25.11.1



Snort experiments:



\- Ubuntu 24.04.4 LTS

\- Snort 2.9.20 GRE Build 82

\- libpcap 1.10.4

\- PCRE 8.39

\- ZLIB 1.3



A fixed random state of `42` was used where applicable.



\## Datasets



The raw CICIDS2017 and UNSW-NB15 datasets are not redistributed in this repository.



Users should obtain the datasets from their official sources. Large dataset files, packet captures and raw Snort alert logs are excluded through `.gitignore`.



\## Results



Selected CSV, JSON, confusion-matrix and figure outputs used to support the thesis findings are included under the `results` directory.



\## Author



Tasos Louka  

MSc Applied Cyber Security  

Technological University Dublin

