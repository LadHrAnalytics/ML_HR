import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Optional, List
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


class RenameDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RenameDataTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_new = X.copy()
        df_new.rename(columns={'income_name': 'income__name'}, inplace=True)
        return df_new


class ProcessSkillsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, freq_cutoff: int = 100,
                       add_unprocessed: bool = False) -> None:
        self.freq_cutoff = freq_cutoff
        self.add_unprocessed = add_unprocessed

        self.morph = pymorphy2.MorphAnalyzer()
        self.nltk_stop_words = set(stopwords.words('russian'))
        self.custom_words = {',', ':', 'работа', 'активный', 'язык', 'навык',
                             'деловой', 'ведение', 'проведение', 'презентация', 'грамотный'}
        self.stop_words = self.nltk_stop_words.union(self.custom_words)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ProcessSkillsTransformer':
        return self

    def tokenize_and_lemmatize(self, skill_list: List[str]) -> List[str]:
        tokens = []
        for skill in skill_list:
            words = word_tokenize(skill)
            lemmas = [self.morph.parse(word.lower())[0].normal_form for word in words if word.lower() not in self.stop_words]
            tokens.extend(lemmas)
        return tokens

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_processed = X.copy()

        def prepare_skills(df: pd.DataFrame, col: str) -> pd.DataFrame:
            df_new = df.copy()
            df_new[col] = df_new[col].apply(self.tokenize_and_lemmatize)
            counts = df_new[col].explode().value_counts()
            frequent = counts[counts >= self.freq_cutoff].index
            df_new[col] = df_new[col].apply(lambda skills: [skill for skill in skills if skill in frequent])
            return df_new

        if self.add_unprocessed:
            df_processed = prepare_skills(df_processed, 'skills')
            df_processed = prepare_skills(df_processed, 'unconverted_skills')
            df_processed['skills_plus'] = df_processed['skills'] + df_processed['unconverted_skills']
        else:
            df_processed['skills_plus'] = df_processed['skills']

        df_processed = df_processed[df_processed['skills_plus'].apply(lambda x: len(x) > 0)]
        df_processed = df_processed.drop(['unconverted_skills', 'skills'], axis=1)
        return df_processed



class VectorizePCASkillsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_components: int = 100,
                 PCA_enable: bool = True,
                 add_unprocessed: bool = False) -> None:

        self.n_components = n_components
        self.PCA_enable = PCA_enable
        self.add_unprocessed = add_unprocessed
        self.vectorizer = None
        self.pca = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VectorizePCASkillsTransformer':
        self.vectorizer.fit(X['skills_plus'])
        if self.PCA_enable:
            skills_tfidf = self.vectorizer.transform(X['skills_plus'])
            skills_df = pd.DataFrame(skills_tfidf.toarray(), columns=self.vectorizer.get_feature_names_out())
            self.pca.fit(skills_df)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_vect = X.copy()

        if not self.add_unprocessed:
            self.vectorizer = TfidfVectorizer(analyzer=lambda x: x)
        else:
            df_vect['skills_plus'] = df_vect['skills_plus'].apply(lambda x: ' '.join(x))
            self.vectorizer = TfidfVectorizer()

        skills_tfidf = self.vectorizer.transform(df_vect['skills_plus'])
        skills_df = pd.DataFrame(skills_tfidf.toarray(), columns=self.vectorizer.get_feature_names_out())
        df_vect = df_vect.reset_index()

        if self.PCA_enable:
            self.pca = PCA(n_components=self.n_components)
            skills_pca = self.pca.transform(skills_df)
            skills_pca_df = pd.DataFrame(skills_pca, columns=[f'component_{i+1}' for i in range(skills_pca.shape[1])])
            df_merged = pd.concat([df_vect, skills_pca_df], axis=1)
        else:
            df_merged = pd.concat([df_vect, skills_df], axis=1)

        df_merged = df_merged.drop(['skills_plus', 'index'], axis=1)
        df_merged[['salary_from_gross', 'salary_to_gross', 'salary_average']] = df_merged[['salary_from_gross', 'salary_to_gross', 'salary_average']].astype(float)

        return df_merged


class ExtractFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_param: Optional[str] = None) -> None:
        self.drop_param = drop_param
        self.transformers = [
            ('schedule_name', OneHotEncoder(sparse_output=False, drop=self.drop_param), ['schedule__name']),
            ('grade', OneHotEncoder(sparse_output=False, drop=self.drop_param), ['grade']),
            ('income_name', OneHotEncoder(sparse_output=False, drop=self.drop_param), ['income__name']),
            ('area_name', OneHotEncoder(sparse_output=False, drop=self.drop_param), ['area__name'])
        ]
        self.pipeline = Pipeline(steps=[
            ('column_transformer', ColumnTransformer(transformers=self.transformers, remainder='passthrough'))
        ])

    def fit(self, X: pd.DataFrame, y=None) -> 'ExtractFeaturesTransformer':
        self.pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_new = X.copy()
        df_new = self.pipeline.transform(df_new)

        column_names = self.pipeline.named_steps['column_transformer'].get_feature_names_out()

        strings_to_replace = ['remainder__','schedule__name_','income__name_', 'area__name_']
        for string in strings_to_replace:
            column_names = [x.replace(string, '') for x in column_names]
        column_names = [x.replace('grade__grade_', 'grade_') for x in column_names]

        df_new = pd.DataFrame(df_new, columns=column_names)

        return df_new