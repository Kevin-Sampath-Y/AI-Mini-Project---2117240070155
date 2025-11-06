# AI-Mini-Project---2117240070155

# Sentiment Analysis Using Naive Bayes Algorithm

## AI Lab Mini Project - Semester III

**Author**: Y Kevin Sampath  
**Register Number**: 2117240070155 
**Department**: Artificial Intelligence and Data Science  
**Institution**: Rajalakshmi Institute of Technology

---

## 📋 Project Overview
This project implements a Sentiment Analysis system using the Naive Bayes algorithm to classify text reviews as positive or negative.

## 🎯 Objectives
- Develop an automated sentiment classification system
- Achieve high accuracy using Naive Bayes algorithm
- Process and analyze real-world text data

## 🛠️ Technologies Used
- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- TfidfVectorizer

## 📊 Dataset
- **Total Records**: 1000 reviews
- **Positive Reviews**: 520
- **Negative Reviews**: 480
- **Source**: Custom generated dataset

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Steps
1. Clone this repository
```bash
git clone https://github.com/[your-username]/AI_MINIPROJECT_[your-reg-no].git
```

2. Navigate to project directory
```bash
cd AI_MINIPROJECT_[your-reg-no]
```

3. Generate dataset (first time only)
```bash
python dataset_generator.py
```

4. Run the main program
```bash
python Sentiment_Analysis.py
```

## 📈 Results
- **Accuracy**: ~87-90%
- **Model**: Multinomial Naive Bayes
- **Feature Extraction**: TF-IDF Vectorization

## 📝 Project Structure

AI_MINIPROJECT_2117240070155/
│
├── Sentiment_Analysis.py          # Main implementation
├── dataset_generator.py           # Dataset creation script
├── sentiment_dataset.csv          # Dataset file
├── sentiment_model.pkl            # Trained model (generated)
├── tfidf_vectorizer.pkl          # Vectorizer (generated)
├── Mini_Project_Report.pdf       # Complete project report
└── README.md                     # This file

## 🔮 Future Enhancements
- Multi-class sentiment classification (positive, negative, neutral)
- Real-time social media sentiment analysis
- Web interface using Flask/Streamlit
- Deep learning implementation (LSTM, BERT)

## 📚 References
1. Scikit-learn Documentation
2. Natural Language Processing with Python
3. Kaggle Sentiment Analysis Datasets

## 👨‍💻 Contact
- **GitHub**: [Your GitHub Username]
- **Email**: [Your Email]

---

**Faculty In-charge**: Mrs. Phebe Persis  
**Course**: Artificial Intelligence Laboratory  
**Academic Year**: 2025-2026
