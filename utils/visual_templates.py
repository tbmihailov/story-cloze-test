import re

import numpy as np

from sklearn import preprocessing


class UIColors_Consts(object):

    flatui_colors = [  # https://flatuicolors.com/
        "rgba(26, 188, 156, 1.0)",
        "rgba(46, 204, 113, 1.0)",
        "rgba(52, 152, 219, 1.0)",
        "rgba(155, 89, 182, 1.0)",
        "rgba(52, 73, 94, 1.0)",

        "rgba(22, 160, 133, 1.0)",
        "rgba(39, 174, 96, 1.0)",
        "rgba(41, 128, 185, 1.0)",
        "rgba(142, 68, 173, 1.0)",
        "rgba(44, 62, 80, 1.0)",

        "rgba(241, 196, 15, 1.0)",
        "rgba(230, 126, 34, 1.0)",
        "rgba(231, 76, 60, 1.0)",
        "rgba(236, 240, 241, 1.0)",
        "rgba(149, 165, 166, 1.0)",
        
        "rgba(243, 156, 18, 1.0)",
        "rgba(211, 84, 0, 1.0)",
        "rgba(192, 57, 43, 1.0)",
        "rgba(189, 195, 199, 1.0)",
        "rgba(127, 140, 141, 1.0)"
    ]

    material_colors = [  #
        "rgba(244, 67, 54, 1.0)",
        "rgba(233, 30, 99, 1.0)",
        "rgba(156, 39, 176, 1.0)",
        "rgba(103, 58, 183, 1.0)",
        "rgba(63, 81, 181, 1.0)",

        "rgba(33, 150, 243, 1.0)",
        "rgba(3, 169, 244, 1.0)",
        "rgba(0, 188, 212, 1.0)",
        "rgba(0, 150, 136, 1.0)",
        "rgba(76, 175, 80, 1.0)",

        "rgba(139, 195, 74, 1.0)",
        "rgba(205, 220, 57, 1.0)",
        "rgba(255, 235, 59, 1.0)",
        "rgba(255, 193, 7, 1.0)",
        "rgba(255, 152, 0, 1.0)",

        "rgba(255, 87, 34, 1.0)",
        "rgba(121, 85, 72, 1.0)",
        "rgba(158, 158, 158, 1.0)",
        "rgba(96, 125, 139, 1.0)",
        "rgba(255, 255, 255, 1.0)"
    ]

def scale(lst, scale_range=[0, 100]):
    scaler = preprocessing.MinMaxScaler(scale_range)
    arr = np.expand_dims(np.asarray(lst),-1)
    arr = scaler.fit_transform(arr)

    arr = np.squeeze(arr)
    return arr

def item_attentions_html_content(item_meta, item_meta_know, item_att, t_id2w, kn_id2w):
    """
    Display data as HTML
    :param item_meta: 
    :param item_meta_know: 
    :param item_att: 
    :param t_id2w: 
    :param kn_id2w: 
    :return: 
    """
    def gen_tokens_att_content(token_att_tuples):
        html_color = "rgba(0,150,50,%s)"
        html_tokens = [("<span style=\"background-color:" + html_color + "\">%s</span>") % (x[1], x[0]) for x in
                       token_att_tuples]

        html_content = "<p>\n" + ' '.join(html_tokens) + "\n</p>\n"
        return html_content

    # print item_att["story_to_question_att"]
    story_to_q_att = np.squeeze(np.asarray(item_att["story_to_question_att"][:len(item_meta["context"]["tokens"])]))

    att_scaled = scale(story_to_q_att, [0, 1.0])
    tokens_attenntion_tuples = [(t_id2w[x], att_scaled[i]) for i, x in enumerate(item_meta["context"]["tokens"])]

    tokens_attention_content = gen_tokens_att_content(tokens_attenntion_tuples)
    question_tokens = [t_id2w[x] for x in item_meta["question"]["tokens"]]
    candidates_tokens = [t_id2w[x] for x in item_meta["candidates"]]
    gold_answer_token = candidates_tokens[item_meta["answer_id"]]
    html = "<table>\n"
    html += "<tr><td><b>Story to quesiton:</b></td></tr>\n"
    html += "<tr><td>"
    html += tokens_attention_content
    html += "</td></tr>\n"
    # quesiton
    html += "<tr><td><b>Question:</b></td></tr>\n"
    html += "<tr><td>" + " ".join(question_tokens) + "</td></tr>\n"

    # quesiton
    html += "<tr><td><b>Candidates:</b></td></tr>\n"
    html += "<tr><td>" + " | ".join(candidates_tokens) + "</td></tr>\n"

    # quesiton
    html += "<tr><td><b>Gold Answer:</b></td></tr>\n"
    html += "<tr><td>" + gold_answer_token + "</td></tr>\n"
    # close table
    html += "</table>\n"

    return html

def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless',
        '>': r'\textgreater',
    }
    regex = re.compile('|'.join(re.escape(unicode(key)) for key in sorted(conv.keys(), key = lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)
