# AI Spam Message Analyzer

A beginner-friendly Python and Streamlit project that estimates whether an SMS
message is spam or not spam. It uses TF-IDF text features and a Multinomial
Naive Bayes classifier trained on the labeled [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

> **Important:** This is an educational tool, not a security product. A model
> confidence score is an estimate, not a guarantee that a message is safe.

## Project files

```text
ai-spam-message-analyzer/
├── app.py                 # Streamlit user interface
├── train_model.py         # Downloads data, trains and evaluates the model
├── requirements.txt       # Python package dependencies
├── README.md              # This guide
├── .gitignore
├── data/
│   └── .gitkeep           # Dataset is downloaded here
└── models/
    └── .gitkeep           # Trained files are saved here
```

The training script downloads the dataset to
`data/SMSSpamCollection.tsv` the first time it runs. It saves the classifier
and vectorizer as `models/spam_classifier.joblib` and
`models/tfidf_vectorizer.joblib`. These generated files are ignored by Git
and can be recreated by training again.

## Requirements

- Windows 10 or 11
- Python 3.9 or newer (Python 3.11 or 3.12 is recommended)
- Visual Studio Code (optional, but useful for editing)
- Internet access for installing packages and the first dataset download

## 1. Install Python and configure VS Code

1. Check whether Python is installed by opening **PowerShell** or the VS Code
   terminal and running:

   ```powershell
   py --version
   python --version
   ```

   A version such as `Python 3.12.x` means Python is installed. If Windows says
   Python was not found or opens the Microsoft Store, install Python from
   [python.org/downloads](https://www.python.org/downloads/). In the installer,
   select **Add python.exe to PATH**, then finish the installation and reopen
   the terminal.

2. In VS Code, open this project folder using **File > Open Folder**.
3. Install Microsoft's **Python** extension from the Extensions view if it is
   not already installed.
4. Press `Ctrl+Shift+P`, run **Python: Select Interpreter**, and select the
   Python installation you just installed. Later, after creating the virtual
   environment below, select the interpreter whose path ends with
   `.venv\Scripts\python.exe`.
5. Open a new VS Code terminal with **Terminal > New Terminal**. Check the
   selected interpreter:

   ```powershell
   python --version
   ```

   If `python` still opens the Microsoft Store, use **Python: Select
   Interpreter** to choose the installed interpreter. For a standard per-user
   Python 3.12 installation, its path is often
   `%LOCALAPPDATA%\Programs\Python\Python312\python.exe`. You can create the
   virtual environment with that interpreter directly:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
   ```

## 2. Create an environment and install dependencies

Run these commands in PowerShell from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When the environment is active, choose `.venv\Scripts\python.exe` as the
interpreter in VS Code using **Python: Select Interpreter**.

If `py` is not recognized but `python --version` works, use
`python -m venv .venv` instead of `py -m venv .venv`.

If PowerShell reports that script execution is disabled, allow activation for
this terminal session only, then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, in **Command Prompt** use:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Download/configure the dataset and train

With the virtual environment active, from the project folder run:

```powershell
python train_model.py
```

The script downloads the public UCI SMS Spam Collection archive into the
`data` folder if `data\SMSSpamCollection.tsv` is not already present. It then
cleans the labeled messages, makes a stratified 80% training / 20% test split,
fits TF-IDF on training text, trains Multinomial Naive Bayes, and prints
accuracy, spam precision, spam recall, spam F1-score, and a classification
report. Finally, it saves the model and vectorizer in `models`.

If the automatic download fails, download the UCI SMS Spam Collection from the
[UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
extract the file named `SMSSpamCollection`, and put it in the project's
`data` folder as `SMSSpamCollection.tsv`. The file must contain tab-separated
rows: label (`ham` or `spam`), followed by message text. Then run
`python train_model.py` again.

Training is needed once, or again whenever you want to rebuild the model. The
Streamlit app loads the saved model files and does not retrain on each launch.

### Verified training results

This project was trained successfully with Python 3.12.10 using the downloaded
UCI dataset. After removing duplicate labeled messages, 5,158 messages remained;
the held-out test set contained 1,032 messages. The observed spam-class
metrics were:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9554 |
| Spam precision | 1.0000 |
| Spam recall | 0.6406 |
| Spam F1-score | 0.7810 |

Metrics can vary if the dataset or training settings change. The recall score
means this model missed some spam in the test set, so do not treat a "Not Spam
Message" prediction as proof of safety.

## 4. Run and test the Streamlit app

After training succeeds, start the app from the project folder:

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal, usually
`http://localhost:8501`. Enter a message and select **Analyze message**.
The page shows **Spam Message** or **Not Spam Message**, an estimated
confidence percentage, and a warning for spam predictions.

Try these examples:

**Likely spam**

```text
Congratulations! You have won a free prize. Call now to claim your reward!
```

**Likely not spam**

```text
Hi, are we still meeting for lunch at 12:30? Let me know when you arrive.
```

Predictions are estimates and can be wrong. In particular, **Not Spam
Message** does not prove a message is trustworthy.

Both sample messages were tested in the running Streamlit interface: the spam
example displayed **Spam Message** with an 82.9% model estimate and a warning;
the lunch message displayed **Not Spam Message** with a 99.5% model estimate.

To stop Streamlit, focus the terminal running it and press `Ctrl+C`.

## Troubleshooting

- **`python` opens the Microsoft Store or isn't found:** install Python from
  python.org, enable **Add python.exe to PATH**, reopen VS Code, and select
  the interpreter using **Python: Select Interpreter**. You can also try
  `py --version`.
- **`py` isn't recognized:** use `python -m venv .venv` if
  `python --version` works.
- **PowerShell won't activate `.venv`:** use the session-only execution policy
  command above or activate from Command Prompt.
- **Dataset download fails:** check internet access or use the manual UCI
  download steps above.
- **The app says model files are missing:** activate `.venv`, run
  `python train_model.py`, and then start `streamlit run app.py`.
- **A package is missing:** activate `.venv` and rerun
  `pip install -r requirements.txt`.

## Before uploading to GitHub

After installing Python, successfully train the model, start the app, and test
both sample messages. Review the files you plan to publish. `.gitignore`
excludes the virtual environment, downloaded dataset, and generated model
artifacts; each user can download the public dataset and train locally.
