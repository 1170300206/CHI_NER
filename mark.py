import pandas as pd


def get_string(x):
    now = x.split('\n')
    o = now[1].split(' ')
    while '' in o:
        o.remove('')
    return o[1]


def get_number(x):
    now = x.split('\n')
    o = now[1].split()
    while '' in o:
        o.remove('')
    if len(o) == 1:
        return -1
    else:
        return int(o[1])


if __name__ == '__main__':
    labels = pd.read_csv('Train_labels.csv')
    reviews = pd.read_csv('Train_reviews.csv')
    out = open('mark.txt', 'w', encoding='utf-8')
    n = 6633
    m = 3229
    id = 1
    i = 0
    while i < n:
        o_l = []
        o_r = []
        a_l = []
        a_r = []
        sum = 0
        while get_number(str(labels.loc[i:i, ['id']])) == id:
            sum += 1
            o_l.append(get_number(str(labels.loc[i:i, ['O_start']])))
            o_r.append(get_number(str(labels.loc[i:i, ['O_end']])))
            a_l.append(get_number(str(labels.loc[i:i, ['A_start']])))
            a_r.append(get_number(str(labels.loc[i:i, ['A_end']])))
            i += 1
            if i >= n:
                break
        string = get_string(str(reviews.loc[id-1:id-1, ['Reviews']]))
        #print(string)
        length = len(string)
        for j in range(length):
            flag = 'O'
            for k in range(sum):
                if flag != 'O':
                    break
                if j == o_l[k]:
                    flag = 'B-OPI'
                elif (j > o_l[k]) and (j < o_r[k]):
                    flag = 'I-OPI'
                elif j == a_l[k]:
                    flag = 'B-ASP'
                elif (j > a_l[k]) and (j < a_r[k]):
                    flag = 'I-ASP'
            out.write(string[j] + ' ' + flag + '\n')
        out.write('end\n')
        #out.write('\n')
        id += 1