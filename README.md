# SDR Dashboard Docker Deployment

This project runs as a Streamlit app. The Docker setup keeps the existing pinned Python package versions from `requirements.txt` and uses the same app entrypoint as the existing `Procfile`.

## 1. Build and run locally

Install Docker Desktop, then run:

```sh
docker compose up --build
```

Open:

```text
http://localhost:8509
```

To use a different host port:

```sh
HOST_PORT=8080 docker compose up --build
```

Then open `http://localhost:8080`.

## 2. Stop the app

Press `Ctrl+C` in the terminal running Docker Compose, then run:

```sh
docker compose down
```

## 3. Build a standalone image

```sh
docker build -t sdr-dashboard:latest .
```

Run it with:

```sh
docker run --rm -p 8509:8501 sdr-dashboard:latest
```

## 4. Deploy to a cloud Docker host

Use any host that can run Docker containers, such as a VM, container service, or app platform. The container listens on `0.0.0.0` and uses `PORT`, defaulting to `8501`.

Typical steps:

1. Push this repository to your Git provider.
2. Configure the cloud service to build from the `Dockerfile`.
3. Expose container port `8501`, or set the platform's `PORT` variable if it assigns one.
4. Deploy the container.

The parameter workbook is included at `data/SDR Parameters.xlsx`, and the container sets:

```text
SDR_PARAMS_PATH=/app/data/SDR Parameters.xlsx
```

If you replace the workbook later, rebuild and redeploy the image so the container includes the new file.
