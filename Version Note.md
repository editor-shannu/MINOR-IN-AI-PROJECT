📄 Project Tools, Libraries, and Version Notes

This project was developed and tested using the following environment and library versions. These versions were selected to ensure compatibility between data processing, time-series forecasting, and Streamlit deployment.

🐍 Programming Language

Python 3.10

Python 3.10 was chosen because:

Streamlit Cloud fully supports it

Compatible with key libraries used in the project

📦 Core Libraries
Frontend / Dashboard
| Library   | Version                  | Purpose                       |
| --------- | ------------------------ | ----------------------------- |
| streamlit | latest (Streamlit Cloud) | Interactive dashboard UI      |
| pandas    | 2.x                      | Data loading and manipulation |
| numpy     | 1.23.x                   | Numerical operations          |

Sentiment Analysis
| Library      | Version | Purpose                  |
| ------------ | ------- | ------------------------ |
| transformers | 4.x     | BERT model for sentiment |
| torch        | 2.3.0   | Backend for transformers |

Forecasting
| Library           | Version           | Purpose                             |
| ----------------- | ----------------- | ----------------------------------- |
| neuralprophet     | 0.5.2             | Time-series forecasting             |
| pytorch-lightning | 1.6.5             | Required by neuralprophet           |
| holidays          | 0.14.x compatible | Required by neuralprophet internals |


Note: NeuralProphet required specific version alignment with PyTorch Lightning and holidays to avoid compatibility issues.

🛠 Development Tools
| Tool                         | Version / Platform |
| ---------------------------- | ------------------ |
| Visual Studio Code           | latest             |
| Git                          | version control    |
| GitHub                       | repository hosting |
| Streamlit Cloud              | deployment         |
| Virtual Environment (`venv`) | used locally       |

🌐 Deployment Settings (Streamlit Cloud)

Runtime: Python 3.10

Required files included:

app.py

combined.csv

sentiment_yearly.csv

requirements.txt

requirements.txt contains only minimal dashboard dependencies to ensure fast deployment.
