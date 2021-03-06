
"""Arabic News Summary"""

from __future__ import absolute_import, division, print_function

import csv

import datasets
from arabert.preprocess import ArabertPreprocessor

model_name="bert-base-arabert"
arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)

_DESCRIPTION = """\
The Arabic news summary contain ~26K samples of open domain title and content.
"title" refers to the summary
"content" refers to the article content
"""

_URL = "https://raw.githubusercontent.com/SalehShmali/Arabic_News_Summarization/main/articles.csv"


class ArabicNewsConv(datasets.GeneratorBasedBuilder):
    """ A Dataset of Topic-Oriented  News Summarization"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="arabic_news_conv",
            version=datasets.Version("1.0.0"),
            description="Full training set",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "title": datasets.Value("string"),
                    "content": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/SalehShmali/Arabic_News_Summarization",
        )

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_file},
            ),
        ]

    def _generate_examples(self, filepath):
        with open("articles.csv",'r',encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for i , row in enumerate(csv_reader):
                if i==0:
                    continue
                yield i, {
                    "title": arabert_prep.preprocess(row[2]),
                    "content": arabert_prep.preprocess(row[5]),
                }
