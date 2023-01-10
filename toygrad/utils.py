def softmax(outputs):
    # Softmax
    for i in range(len(outputs)):
        for j in range(len(outputs[0])):
            outputs[i][j] = outputs[i][j].exp()
        e_sum = sum(outputs[i])
        for j in range(len(outputs[0])):
            outputs[i][j] = outputs[i][j] / e_sum
    return outputs


def cross_entropy(y_scores, y_true):
    return -sum([y_scores[i][y].log() for i, y in enumerate(y_true)])


def argmax(lst):
    return lst.index(max(lst))
