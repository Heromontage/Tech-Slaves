# Tech-Slaves — SentinelFlow

A FastAPI backend paired with a static HTML/JS frontend for network analysis and optimization.

---

## Repository layout

```
Tech-Slaves/
├── backend/          # Python / FastAPI application
│   ├── main.py
│   ├── requirements.txt
│   ├── docker-compose.yml   # PostgreSQL + Neo4j
│   └── env.example
└── site/
    └── public/       # Static frontend (open directly in a browser)
```

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| Python | 3.11 | Backend runtime |
| Docker Desktop | latest | Runs PostgreSQL & Neo4j |
| Git | any | Clone the repo |
| A browser | any | Serve the frontend |

> **Windows users:** Docker Desktop must have the WSL 2 backend enabled (default on modern installs).

---

## 1 — Start on WSL (Ubuntu / Debian)

### 1.1 Clone the repository

```bash
git clone https://github.com/Heromontage/Tech-Slaves.git
cd Tech-Slaves
```

### 1.2 Start the databases with Docker

```bash
cd backend
docker compose up -d
```

This spins up:
- **PostgreSQL** on `localhost:5432`
- **Neo4j** on `localhost:7687` (Browser UI at `http://localhost:7474`)

### 1.3 Configure environment variables

```bash
cp env.example .env
```

Open `.env` and change the host values so they point to `localhost` (the defaults point to Docker service names):

```dotenv
POSTGRES_HOST=localhost
NEO4J_URI=bolt://localhost:7687
```

Leave all other values at their defaults for a local dev setup.

### 1.4 Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> PyTorch is included in `requirements.txt` with a CPU-only extra index URL, so no GPU is required.

### 1.5 Apply database migrations

```bash
alembic upgrade head
```

### 1.6 Start the backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 1.7 Open the frontend

Open a new terminal, navigate to `site/public`, and serve the files with Python's built-in HTTP server:

```bash
cd site/public
python3 -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

---

## 2 — Start on Windows (PowerShell / Command Prompt)

### 2.1 Clone the repository

```powershell
git clone https://github.com/Heromontage/Tech-Slaves.git
cd Tech-Slaves
```

### 2.2 Start the databases with Docker

Open Docker Desktop and make sure it is running, then in PowerShell:

```powershell
cd backend
docker compose up -d
```

This spins up:
- **PostgreSQL** on `localhost:5432`
- **Neo4j** on `localhost:7687` (Browser UI at `http://localhost:7474`)

### 2.3 Configure environment variables

```powershell
Copy-Item env.example .env
```

Open `.env` in a text editor and update the host values:

```dotenv
POSTGRES_HOST=localhost
NEO4J_URI=bolt://localhost:7687
```

### 2.4 Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.5 Apply database migrations

```powershell
alembic upgrade head
```

### 2.6 Start the backend

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 2.7 Open the frontend

Open a second PowerShell window, navigate to `site\public`, and serve the files:

```powershell
cd site\public
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

---

## Stopping everything

```bash
# Stop the backend — press Ctrl+C in the terminal running uvicorn

# Stop the databases
cd backend
docker compose down
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` when starting uvicorn | Make sure the virtual environment is activated (`source .venv/bin/activate` / `.venv\Scripts\activate`) |
| Database connection refused | Check that `docker compose up -d` completed successfully (`docker compose ps`) |
| CORS errors in the browser | Ensure the backend is running on port `8000` (CORS is pre-configured for ports 3000, 5173, and 8080) |
| Port already in use | Change the port: `uvicorn main:app --port 8001` |
| Neo4j auth error | Verify `NEO4J_PASSWORD` in `.env` matches the value set in `docker-compose.yml` (default: `sentinelflow`) |
