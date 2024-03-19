### Run it locally

1. Clone the repository
2. Run the following commands:

```bash
poetry install
poetry run python -m main --alpha=0.1 --l1_ratio=0.7
poetry run mlflow ui // In another terminal
```

or

```bash
poetry install
poetry shell
python -m main --alpha=0.1 --l1_ratio=0.7
mlflow ui // In another terminal
```

## MLFlow configurations
Localhost ![MLFlow configurations](docs/.attachments/tracking_server1.png)
Localhost with db for parameters ![MLFlow configurations](docs/.attachments/tracking_server2.png)
Localhost Tracking Server ![MLFlow configurations](docs/.attachments/tracking_server3.png)
Remote host ![MLFlow configurations](docs/.attachments/tracking_server4.png)
Proxy server ![MLFlow configurations](docs/.attachments/tracking_server1.png)
Proxy server just for arctifacts ![MLFlow configurations](docs/.attachments/tracking_server1.png)