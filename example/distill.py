from zeno import distill, DistillReturn


@distill
def subject(df, ops):
    ret = []
    for i, row in df.iterrows():
        ret.append(row[ops.data_column][1]["content"].split("\n")[0][9:])
    return DistillReturn(distill_output=ret)
