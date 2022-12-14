# #!pip install pip --upgrade
# #!pip install pyopenssl --upgrade
#!pip install bert-extractive-summarizer
#!!pip install rouge-score
import fitz
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from nltk import tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from summarizer import Summarizer
from rouge_score import rouge_scorer
class TextSumerizationPDF():
    def __init__(self,name_pdf):
        self.name_pdf = name_pdf
        self.abstract_type = ['Abstract', 'ABSTRACT']
        self.all_text, self.abstract_page, self.all_blocks = self.get_data_from_pdf()
        self.text_to_preprocessing, self.orginal_abstract = self.remove_abstract_from_pdf()
        self.text_after_reprocessing = self.pdf_preprocessing()
        self.model_abstract = self.bert_extractive_summarizer()
        print(self.model_abstract)
        self.scores = self.evaluation()
        print(self.scores)
        self.df_blocks = self.get_data_from_blocks()

    # -------------------------------------------------------------- get data
    def get_data_from_pdf(self):
        all_blocks, abstract_page, all_text = [], [], ''
        for page in fitz.open(self.name_pdf):
            all_text += page.get_text()
            all_blocks.append(page.get_text("blocks"))
            if any(ext in page.get_text() for ext in self.abstract_type):
                abstract_page.append(page.get_text())
        return all_text, abstract_page[0], all_blocks

    def remove_abstract_from_pdf(self):
        split_page = self.abstract_page.split('.')
        abstract_sent = [sentence + '.' for sentence in split_page if any(ext in sentence for ext in self.abstract_type)][0]
        abstract_sent = [abstract_sent[abstract_sent.find(x):] for x in self.abstract_type][0]
        keywords_sent = [sentence + '.' for sentence in split_page if 'Keywords' in sentence][0]
        keywords_sent = keywords_sent[keywords_sent.find('Keywords'):]
        data_batween = self.abstract_page[self.abstract_page.rfind(abstract_sent)+len(abstract_sent):self.abstract_page.rfind(keywords_sent)]
        orginal_abstract = abstract_sent + data_batween + keywords_sent
        text_to_preprocessing = self.all_text.replace(orginal_abstract, '')
        return text_to_preprocessing, orginal_abstract
    #-------------------------------------------------------------- preprocessing
    def data_preprocessing(self, row):
        # Lowering letters
        row = row.str.lower()
        # Removing html tags
        row = row.str.replace(r'<[^<>]*>', '', regex=True)
        # Removing urls
        row = row.str.replace(r'http\S+', '', regex=True)
        # Removing numbers
        row = row.str.replace('\d+', '')
        # Remove special characers
        row = row.str.replace('\W', ' ')
        # Distribution of tokens
        row = row.apply(word_tokenize)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        row = row.apply(lambda x: [item for item in x if item not in stop_words])
        # Convert lists to strings
        row = row.apply(lambda x: ' '.join(map(str, x)))
        return row

    def pdf_preprocessing(self):
        sentenceSplit = tokenize.sent_tokenize(self.text_to_preprocessing)
        df = pd.DataFrame(sentenceSplit, columns=['Sentence'])
        df["Sentence"] = self.data_preprocessing(df["Sentence"])
        df = pd.DataFrame(['. '.join(df['Sentence'].to_list())], columns=['Sentence'])
        return df.values.tolist()[0][0]

    # -------------------------------------------------------------- models
    def bert_extractive_summarizer(self):
        len_abstract = len(self.orginal_abstract.split('.'))
        model = Summarizer()
        model_abstract = model(self.text_after_reprocessing, num_sentences=len_abstract)
        return model_abstract

    def evaluation(self):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(self.model_abstract, self.orginal_abstract)
        return scores

    # -------------------------------------------------------------- error analysis
    def get_data_from_blocks(self):
        all_data_blocks = [(page_ind,) + tup for page_ind in range(len(self.all_blocks)) for tup in self.all_blocks[page_ind]]
        df_blocks = pd.DataFrame(all_data_blocks, columns=['page', 'x0', 'y0', 'x1', 'y1', 'text', 'paragraph', 'img/text_index'])
        df_blocks = df_blocks[df_blocks["img/text_index"] == 0]
        df_blocks = df_blocks[['page', 'text', 'paragraph']]
        df_blocks["text"] = self.data_preprocessing(df_blocks["text"])
        df_blocks = df_blocks.loc[df_blocks["text"] != '']
        return df_blocks

    def error_analysis(self):
        # To-do
        model_abstract_split = self.model_abstract.split('.')
        self.df_blocks[self.df_blocks.text.str.contains('master thesis')]


if __name__ == "__main__":
    name_pdf = '2.pdf'
    TextSumerizationPDF(name_pdf)