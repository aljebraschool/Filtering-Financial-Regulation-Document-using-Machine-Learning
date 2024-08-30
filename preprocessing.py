from collections import defaultdict
import pandas as pd

def create_relevant_keyword(df):
    all_words = ' '.join(df['Content'].astype(str)).lower().split()
    word_count = defaultdict(int)
    for word in all_words:
        word_count[word] += 1
    return sorted(word_count, key=word_count.get, reverse=True)[:100]


def calculate_keyword_relevant_score(content, keywords):
    content_lower = content.lower()
    return sum(1 for keyword in keywords if keyword in content_lower)


def preprocess_documents(df):

    regulator_score = {regulator: 50 for regulator in df['RegulatorId'].unique()}
    document_type_score = {doc_type: 50 for doc_type in df['DocumentTypeId'].unique()}
    language_scores = {lang: 50 for lang in df['SourceLanguage'].unique()}
    keywords = create_relevant_keyword(df)

    results = []
    for _, doc in df.iterrows():
        content = str(doc['Content']) if pd.notna(doc['Content']) else ''
        title = str(doc['Title']) if pd.notna(doc['Title']) else ''
        combined_text = content + ' ' + title

        regulator_scores = regulator_score.get(doc['RegulatorId'], 50)
        document_type_scores = document_type_score.get(doc['DocumentTypeId'], 50)
        keyword_score = calculate_keyword_relevant_score(combined_text, keywords)
        language_score = language_scores.get(doc['SourceLanguage'], 50)

        # Calculate the combined score
        combined_score = (regulator_scores * 0.25 + document_type_scores * 0.25 + keyword_score * 0.3 + language_score * 0.2)

        results.append({
            'DocumentID': doc['DocumentID'],
            'RegulatorScore': regulator_scores,
            'DocTypeScore': document_type_scores,
            'KeywordScore': keyword_score,
            'LanguageScore': language_score,
            'CombinedScore': combined_score

        })

    return pd.DataFrame(results)

