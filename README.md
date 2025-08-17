# Amazon Product Recommendation System

## 📊 Project Overview

A sophisticated data-driven recommendation system for Amazon products that leverages **571.54 million reviews** across **48 million items** in 33 categories. The system employs a dual-analysis strategy combining product metadata and user reviews to deliver highly personalized product recommendations.

### Key Features
- **Two-Stage Analysis Pipeline**: Metadata analysis → Review analysis
- **Real-time Data Processing**: Apache Kafka integration for streaming
- **Scalable Architecture**: Multi-threaded BFS web crawler with domain diversity optimization
- **Advanced Similarity Matching**: Cosine similarity with LSH (Locality-Sensitive Hashing)
- **High Performance**: ~50 pages/sec throughput, <100MB memory for 100K URLs

## 🏗️ System Architecture

### Components

#### 1. Web Crawler (`webcrawler_BFS`)
- **Multi-threaded Architecture**: 8 concurrent workers
- **Priority Queue**: Domain diversity + depth scoring
- **Bloom Filter**: Memory-efficient deduplication
- **Token Bucket Rate Limiter**: Per-host politeness control
- **Thread-safe Design**: 9 fine-grained locks
- **Robots.txt Compliance**: Cached with TTL

#### 2. Similarity Analysis Engine
- **Data Pipeline**:
  ```
  Read Data → Tokenization → Stop Word Removal → HashingTF 
  → Normalization → LSH → Similarity Calculation → Recommendation
  ```
- **Features Extracted**: title, features, description, categories, details
- **LSH Configuration**: 
  - bucketLength: 0.5 (adjustable to 0.25 for higher accuracy)
  - Distance threshold: 5

#### 3. Middleware Integration
- **Apache Kafka**: Real-time data streaming backbone
  - Topic-1: Top 50 similar products from metadata
  - Topic-2: Final top 5 recommendations from review analysis
- **ngrok**: Secure tunnel for external access

#### 4. Review Analysis Module
- **Dataset**: 2,475,451 cleaned reviews
- **Verification Strategy**: Includes both verified (93.5%) and unverified (6.5%) reviews
- **Metrics Analyzed**:
  - Rating distributions
  - Helpful votes percentage
  - Product coverage analysis

## 📁 Project Structure

```
Amazon-recommender/
├── crawler/
│   ├── webcrawler_BFS.py       # Main crawler implementation
│   ├── robots_cache.py         # Robots.txt caching module
│   └── redirect_handler.py     # Redirect handling logic
├── similarity/
│   ├── metadata_analysis.py    # Product metadata processing
│   ├── lsh_similarity.py       # LSH implementation
│   └── cosine_similarity.py    # Similarity calculations
├── middleware/
│   ├── kafka_producer.py       # Kafka data producer
│   ├── kafka_consumer.py       # Kafka data consumer
│   └── ngrok_config.py         # ngrok tunnel configuration
├── review_analysis/
│   ├── data_cleaning.py        # Review data preprocessing
│   ├── sentiment_analysis.py   # Review sentiment processing
│   └── ranking_engine.py       # Final recommendation ranking
├── config/
│   ├── crawler_config.json     # Crawler configuration
│   └── kafka_config.json       # Kafka settings
└── data/
    ├── crawl_list*.txt         # Seed URLs
    └── output/                 # Crawled results

```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Apache Spark (for LSH processing)
- Apache Kafka 2.8+
- ngrok (for external access)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YZCUS/Amazon-recommender.git
cd Amazon-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Kafka server:
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties
```

4. Configure ngrok:
```bash
ngrok tcp 9092  # Expose Kafka broker
```

### Usage

1. **Run the web crawler**:
```python
python crawler/webcrawler_BFS.py --config config/crawler_config.json
```

2. **Start metadata analysis**:
```python
python similarity/metadata_analysis.py --input-asin B08R39MRDW
```

3. **Process reviews**:
```python
python review_analysis/ranking_engine.py --top-k 5
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Throughput | ~50 pages/sec |
| Memory Usage | <100MB (100K URLs) |
| Success Rate | >95% |
| Max Crawl Depth | 100 |
| Max Pages | 200 |
| Concurrent Workers | 8 |
| Review Dataset | 2.47M reviews |
| Product Coverage | 48M items |

## 🔧 Configuration

### Crawler Configuration (`crawler_config.json`)
```json
{
  "max_pages": 200,
  "max_depth": 100,
  "num_workers": 8,
  "robots_ttl": 3600,
  "max_redirects": 10,
  "rate_limit": {
    "requests_per_second": 10,
    "per_host": true
  }
}
```

### LSH Parameters
- **numHashTables**: 1 (default, increase for accuracy)
- **bucketLength**: 0.5 (decrease for higher sensitivity)
- **distance_threshold**: 5

## 📊 Key Findings

1. **Review Analysis Insights**:
   - Unverified reviews provide unique perspectives (6.5% of dataset)
   - 51,694 products exclusively reviewed in unverified reviews
   - Similar helpfulness ratings between verified/unverified reviews (57.94% vs 56.53%)

2. **Recommendation Accuracy**:
   - Successfully identifies similar products with different sizes/colors/styles
   - Cosine similarity scores range from 0.502 to 1.0 for top matches

## 🤝 Contributors

- **Zheng-Chen Yao** (zy2876) - Lead Developer
- **Reet Nandy** (rn2528)
- **Jacqueline Ji** (xj235)
- **Yu-Yuan Chang** (yc6549)
- **Tzu-Yi Chang** (tc3930)

## 📄 License

This project is part of NYU Computer Science coursework.

## 🙏 Acknowledgments

- McAuley Lab, UCSD - Amazon review dataset
- NYU Computer Science Department
- Apache Kafka and Spark communities

---

**Note**: This system is designed for educational purposes and research. For production use, additional optimization and scaling considerations are recommended.