



# Socially Aware Spatial MRF Recommender

Production-ready API for top-N recommendation using Socially-Aware Bayesian Markov Model with social trust network integration.

**🌐 Live Demo:** [https://huggingface.co/spaces/PriyankaNidada/recommended-systems](https://priyankanidada-recommended-systems.hf.space/ui)

**📺 Project Video:** [Epinions Recommendation System Demo](https://youtu.be/fTDacdtgvng?si=ICQoBSLk06JArFKt)

---

## 👥 Contributors

- **[Sanjana Garimella](https://github.com/sanjana-garimella/)**
- **[Priyanka Nidadavolu](https://github.com/Priyanka-Nidadavolu)**

---

## 📌 Project Overview

This project implements a **top-N item recommendation system** using a **Socially-Aware Bayesian Markov Model** that leverages a **Markov Random Field (MRF)** to incorporate opinions from socially connected neighbors in a **trust network**. 

The model uses **Bayesian-inspired Logistic Regression** with **35 features**, including:
- **Review characteristics:** price paid, timestamp, review text length
- **Social network features:** trust relationships, PageRank scores, centrality measures
- **Engineered interaction features**

**Key Challenge:** Handles extreme sparsity (99.99%) and cold-start scenarios where traditional collaborative filtering fails.

---

## 🧠 Models Implemented

Three models were compared using **AUC** to evaluate their ability to rank items users would rate highly (≥4 stars):

### 1️⃣ Jaccard Similarity (Baseline)
- Memory-based collaborative filtering
- User-user and item-item similarity
- No content or social features
- **AUC: 0.4997** (essentially random)

### 2️⃣ Bayesian Logistic Regression
- Uses 3 review features: price, time, review text length
- Probabilistic model for rating prediction
- **AUC: 0.5830**

### 3️⃣ Social Bayesian Markov Model ⭐
- Combines Bayesian Logistic Regression (35 features) + MRF social smoothing
- Incorporates trust network to propagate preference signals
- **AUC: 0.6248** (~25% improvement over baseline)

---

## 📂 Dataset

**Epinions Dataset** from Shopping.com consumer reviews:

**Review Data:**
- 50,000 reviews
- 39,719 users
- 11,197 items
- **Sparsity: 99.99%**
- 85% of users have only one review

**Trust Network:**
- 84,516 users
- 1,159,338 directed trust edges
- Mean in/out degree: 13.72

This extreme sparsity makes it ideal for studying **cold-start problems** in recommendation systems.

---

## 📈 Performance Results

| Model                   | AUC Score |
|------------------------|-----------|
| Jaccard Similarity     | 0.4997    |
| Bayesian               | 0.5830    |
| Social Bayesian Markov | **0.6248** |

---

## 📚 Comparison with Literature

Inspired by ["Scalable Recommendation with Social Influence and Sequential Modeling" (IJCAI 2017)](https://cseweb.ucsd.edu/~jmcauley/pdfs/ijcai17a.pdf).

| Model | AUC Score |
|-------|-----------|
| Bayesian (Paper) | 0.53 |
| Sequential Markov + Social Bayesian (Paper) | 0.58 |
| Bayesian (Our Project) | 0.58 |
| **Socially-Aware Spatial Markov (Our Project)** | **0.6248** |

**Key Insight:** Our spatial Markov approach outperforms the paper's sequential model by ~6% for top-N recommendations by better capturing item correlations rather than just sequential patterns.

---

## 🔍 Key Insights

- ✅ Collaborative filtering fails under extreme sparsity (performs like random guessing)
- ✅ Probabilistic models provide significant gains in sparse settings
- ✅ Social trust information dramatically improves recommendation quality
- ✅ Combining content + behavior + social structure yields best performance
- ✅ Spatial Markov modeling > sequential for top-N recommendations

---

## 🚀 Quick Start

### Local Deployment (Free)

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Train models (sample size for quick testing)
python train_and_save.py 5000

# 3. Start the API
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Access:**
- API Docs: http://localhost:8000/docs
- Interactive UI: http://localhost:8000/ui
- Health Check: http://localhost:8000/health

### Docker Deployment (Free)

```bash
docker compose up --build
```

### Hugging Face Spaces (Free)

Already deployed! Visit: https://huggingface.co/spaces/PriyankaNidada/recommended-systems

---

## 🌐 API Endpoints

Once deployed, your API provides:

- **GET /** - API information and available models
- **GET /health** - Health check
- **GET /meta** - Users/items metadata for UI
- **GET /ui** - Interactive web interface
- **POST /recommend** - Get top-k item recommendations
- **POST /predict** - Predict rating probability for user-item pair

### Example Usage

**Get Recommendations:**
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "k": 10,
    "model_type": "social_bayesian"
  }'
```

**Predict Rating:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "item_id": "item456",
    "model_type": "social_bayesian"
  }'
```

**Python Client:**
```python
import requests

# Get recommendations
response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "user_id": "user123",
        "k": 10,
        "model_type": "social_bayesian"
    }
)
recommendations = response.json()
```

---

## 🛠️ Technologies & Tech Stack

**Machine Learning:**
- scikit-learn (Logistic Regression, StandardScaler)
- NetworkX (Graph analysis, PageRank)
- NumPy, Pandas (Data processing)
- Markov Random Fields (MRF)

**Production API:**
- FastAPI (REST API framework)
- Pydantic (Request/response validation)
- Uvicorn (ASGI server)
- joblib (Model persistence)

**Deployment:**
- Docker & Docker Compose
- Hugging Face Spaces
- Port 7860 (HF Spaces standard)

---

## 📁 Project Structure

```
recommended-systems/
├── app.py                    # FastAPI application
├── models.py                 # Model implementations
├── train_and_save.py         # Training script
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── saved_models/            # Trained model artifacts
│   ├── jaccard_model.pkl
│   ├── bayesian_model.pkl
│   ├── social_bayesian_model.pkl
│   └── item_names.json
├── epinions_data/           # Dataset (not included in repo)
│   ├── epinions.txt
│   └── network_trust.txt
└── README.md                # This file
```

---

## 📊 Data Requirements

To train models locally, place dataset files in `epinions_data/`:
- `epinions.txt` - Review data
- `network_trust.txt` - Trust network data

Dataset not included in repository due to size. Pre-trained models are provided in `saved_models/`.

---

## ✨ Features

**Interactive UI:**
- ✅ Dropdown menus for real product names
- ✅ User selection from trained dataset
- ✅ A/B testing dashboard to compare models
- ✅ Real-time probability predictions
- ✅ Top-K recommendations display

**Production-Ready:**
- ✅ Health checks and monitoring
- ✅ Proper error handling
- ✅ Model versioning (numpy 1.24.3, scikit-learn 1.3.0)
- ✅ Dockerized deployment
- ✅ OpenAPI/Swagger documentation
- ✅ CORS enabled for web access

---

## 🎯 Use Cases

- **Cold-start recommendations** for new users with zero history
- **Trust-based filtering** for niche product discovery
- **Social influence modeling** in e-commerce/review platforms
- **Sparse data scenarios** where traditional collaborative filtering fails
- **Research** on social network effects in recommendation systems


---

## 🏆 Achievements

- ✅ **25% improvement** over baseline collaborative filtering
- ✅ **6% improvement** over IJCAI 2017 paper baseline
- ✅ Handles **99.99% sparsity** effectively
- ✅ Production-ready API with **interactive UI**
- ✅ Deployed on **Hugging Face Spaces**

---

## 🤝 Acknowledgments

- Inspired by IJCAI 2017 paper on Social and Sequential Modeling
- Epinions dataset from Shopping.com
- Built as part of academic research project at UC San Diego

