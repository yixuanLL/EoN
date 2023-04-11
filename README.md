# EoN

This package is built based on the paper "Echo of Neighbors: Privacy Amplification for Personalized Private Federated Learning with Shuffle Model".
1. Package privAmp compare several privacy amplification results with shuffle model.
2. Package perS implement the framework APES and S-APES.

# Usage
Privacy amplification in privAmp:
```
python evaluate_bounds.py
```

Framework APES and S-APES in perS:
```
python main.py --optimizer=personalLdpPgd --eps_dist=uniform --per_left=0.1 --per_right=2
```

