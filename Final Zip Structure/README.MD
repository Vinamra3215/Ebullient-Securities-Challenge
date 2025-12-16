# Team 67 - Endterm Submission

## Submission ZIP file structure
```
Team67_H1.zip
├── README.md           #This document
├── requirements.txt    #Requirements
├── Team67_H1.pdf       #Report
├── EBX.py              #Generates signals CSV for EBX
├── EBY.py              #Generates signals CSV for EBY
├── configIIT.json      #Config for backtest
└── mainIIT.py          #Backtest on generated signals
```

## Setup and Run

1. Set config in configIIT.json
```
{
    "data_path": <data directory path>,
    "start_date": 0,
    "end_date": 510,
    "timer": 600,
    "tcost": 2,
    "broadcast": [
        "EBY",
        "EBX"
    ]
}
```

2. Generate Signals
```
python3 EBX.py --data-dir <path to EBX CSVs> 

python3 EBY.py --data-dir <path to EBY CSVs>
```

3. Backtest
```
python mainIIT.py configIIT.json
```