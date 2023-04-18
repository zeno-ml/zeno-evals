from zeno import distill, DistillReturn
from wordfreq import word_frequency


@distill
def number_of_blanks(df, ops):
    ret = []
    for i, row in df.iterrows():
        ret.append(row[ops.data_column][1]["content"].count("_"))
    return DistillReturn(distill_output=ret)


@distill
def answer_length(df, ops):
    ret = []
    for i, row in df.iterrows():
        ret.append(len(row[ops.data_column][1]["content"].split("Letters: ")[1]))
    return DistillReturn(distill_output=ret)


@distill
def word_freq(df, ops):
    ret = []
    for i, row in df.iterrows():
        ret.append(word_frequency(row[ops.label_column], "en") * 1000000)
    return DistillReturn(distill_output=ret)
