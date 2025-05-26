## Project Description

This Presence Detection System leverages face embeddings and Elasticsearch to automate real-time attendance tracking. It analyzes and augments your image database, generates and stores face embeddings (70% in Elasticsearch, 30% cached for threshold validation), and provides both batch threshold-testing scripts and a live video demo. A simple GUI enrollment tool lets you register new users with a webcam, capturing their name and five snapshots to update the Elasticsearch index.  


```markdown
# Presence Detection System

A pipeline for real-time attendance tracking using face embeddings and Elasticsearch.

---

## 📁 Project Structure

```

.
├── analyse.py
├── augmentation.py
├── generate\_and\_store\_embeddings.py
├── real\_time\_first\_attempt.py
├── testing\_on\_threshold/
│   ├── … (threshold experiments)
│   └── final\_test.py
├── cache/                   # cached 30% embeddings
├── db\_infos/                # CSVs with per-person image counts
├── implementation/
│   ├── config.py
│   ├── database.py
│   ├── detection.py
│   ├── recognition.py
│   ├── presence\_logs.py
│   ├── enrollment.py       # GUI for capturing new users
│   └── main.py             # end-to-end demo on local video
└── README.md

````

---

## 🔧 Requirements

- Python 3.8+  
- Docker & Docker Compose  
- Elasticsearch & Kibana (run via Docker)  
- Install Python dependencies:
  ```bash
  pip install -r requirements.txt
````

---

## ⚙️ Installation

1. **Clone & create virtual environment**

   ```bash
   git clone https://github.com/yourusername/presence-detection.git
   cd presence-detection
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Elasticsearch & Kibana**

   ```bash
   docker-compose up -d
   ```

---

## 🚀 Usage

1. **Data Analysis**

   ```bash
   python analyse.py
   ```

   Reads `./db_infos/*.csv`, plots distributions.

2. **Augmentation**

   ```bash
   python augmentation.py
   ```

   Augments individuals with only one image.

3. **Embedding Generation**

   ```bash
   python generate_and_store_embeddings.py
   ```

   * Stores 70% of embeddings in Elasticsearch
   * Caches 30% under `./cache/` for threshold testing

4. **Threshold Testing**

   ```bash
   cd testing_on_threshold
   python final_test.py
   ```

   Produces comparison plots between cached and ES embeddings.

5. **Real-Time Demo**

   * **Prototype (no ES, no threshold):**

     ```bash
     python real_time_first_attempt.py
     ```
   * **OOP Pipeline on local video:**

     ```bash
     python implementation/main.py --video path/to/video.mp4
     ```

6. **Enrollment GUI**

   ```bash
   python implementation/enrollment.py
   ```

   Opens camera GUI to register a new user (name + 5 snapshots → ES).




