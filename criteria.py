import numpy as np


'''
calculate temporal intersection over union
'''
def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))

    if union[1] - union[0] < -1e-5:
        return 0
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    return iou if iou >= 0.0 else 0.0

'''
calculate temporal intersection over union
'''
def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1]-inter[0]
    length = sliding_clip[1]-sliding_clip[0]
    nIoL = 1.0*(length-inter_l)/length
    return nIoL


def nms_temporal(predict_score, predict_windows, overlap):
    pick = list()
    starts = predict_windows[:,0]
    ends = predict_windows[:,1]
    scores = predict_score
    assert len(starts)==len(scores)
    if len(starts)==0:
        return pick

    unions = ends - starts
    indexs = [x[0] for x in sorted(enumerate(scores), key=lambda x:x[1])] # sort and get index

    while len(indexs)>0:
        i = indexs[-1]
        pick.append(i)

        lefts = [max(starts[i],starts[j]) for j in indexs[:-1]]
        rights = [min(ends[i],ends[j]) for j in indexs[:-1]]
        inters = [max(0.0, right-left) for left, right in zip(lefts, rights)]
        laps = [inters[u]/(unions[i] + unions[indexs[u]] - inters[u]) for u in range(len(indexs)-1)]
        indexs_new = []
        for j in range(len(laps)):
            if laps[j] <= overlap:
                indexs_new.append(indexs[j])
        indexs = indexs_new

    return pick

def compute_IoU_recall_top_n(predict_windows, gt_windows, picks, top_n, IoU_thresh):

    correct = 0
    if top_n < len(picks):
        cur_picks = picks[0:top_n]
    else:
        cur_picks = picks
    for index in cur_picks:
        pred_start = predict_windows[index][0]
        pred_end = predict_windows[index][1]
        iou = calculate_IoU(gt_windows, (pred_start, pred_end))
        if iou >= IoU_thresh:
            correct = 1
            break

    return correct

def compute_IoU_recall(predict_score, predict_windows, gt_windows):

    IoU_threshs = [0.1, 0.3, 0.5, 0.7]
    top_n_list = [1]
    topn_IoU_matric = np.zeros([1, 4],dtype=np.float32)

    for i, IoU_thresh in enumerate(IoU_threshs):
        picks = nms_temporal(predict_score, predict_windows, IoU_thresh-0.05)

        for j, top_n in enumerate(top_n_list):
            correct = compute_IoU_recall_top_n(predict_windows, gt_windows, picks, top_n, IoU_thresh)
            topn_IoU_matric[j,i] = correct

    return  topn_IoU_matric

if __name__ == '__main__':

    frame_pred = np.random.rand(200)
    print(frame_pred)
    frame_pred = (frame_pred - np.mean(frame_pred)) / np.std(frame_pred)
    scale = max(max(frame_pred), -min(frame_pred))/0.5
    frame_pred = frame_pred / (scale + 1e-3) + 0.5
    print(frame_pred)
    frame_pred_in = np.log(frame_pred)
    frame_pred_out = np.log(1 - frame_pred)
    candidate_num = 10
    start_end_matrix = np.zeros([200,200], dtype=np.float32)
    start_end_matrix[0, 0] = frame_pred_in[0] + np.sum(frame_pred_out[1:])
    for start in range(200):
        for end in range(200):
            if start == end:
                start_end_matrix[start, end] = frame_pred_in[start] + np.sum(frame_pred_out[:start]) + np.sum(
                    frame_pred_out[end + 1:])
            elif end > start:
                start_end_matrix[start, end] = start_end_matrix[start, end - 1] + frame_pred_in[end] - frame_pred_out[
                    end]
            else:
                start_end_matrix[start, end] = -1e10

    predict_matrix_i = start_end_matrix
    print(predict_matrix_i)

    predict_score = np.zeros([candidate_num], dtype=np.float32)
    predict_windows = np.zeros([candidate_num, 2], dtype=np.float32)

    for cond_i in range(candidate_num):
        #max = np.max(predict_matrix_i)
        idxs = np.where(predict_matrix_i == max)
        start = idxs[0][0]
        end = idxs[1][0]
        print('cond_i:', cond_i)
        print(start)
        print(end)

        predict_score[cond_i] = max
        predict_windows[cond_i, :] = [start, end]
        predict_matrix_i[start, end] = -1e11
        print(predict_windows)

    print(predict_score)
    print(predict_windows)

    # a = np.array([1, 4, 3, 2, 2.5])
    # b = np.array([[0, 3], [0, 5], [2, 5], [3, 5], [4, 5]])
    # c = np.array([4.2,5])
    # p = nms_temporal(a,b,0.5)
    # res = compute_IoU_recall(a,b,c)
    # print(p)
    # print(res)