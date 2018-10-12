import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, id_track, starting_frame=None):
        print('track ' + str(id_track) + ' - create')
        self.id = id_track
        self.starting_frame = starting_frame
        self.age = 0
        self.nframes_missing = 0
        self.detections = []

    def add_detection(self, detection):
        print('track ' + str(self.id) + ' - add detection')
        self.age += 1
        self.nframes_missing = 0
        self.detections.append(detection)

    def mark_missing_frame(self):
        print('track ' + str(self.id) + ' - missing frame')
        self.age += 1
        self.nframes_missing += 1
        self.detections.append('missing')

    def get_last_detection(self):
        for i in range(len(self.detections) - 1, -1, -1):
            if not self.detections[i] == 'missing':
                return self.detections[i]
        raise Exception('Tracks with no detections.')

    def get_nframes_missing(self):
        return self.nframes_missing

    def get_id(self):
        return self.id


class Tracker:
    def __init__(self):
        self.tracks = []
        self.threshold_iou = 0.5
        self.threshold_discard_track = 3
        self.track_count = 0
        self.frame_count = 0

    def create_track(self, detection):
        self.track_count += 1
        track = Track(self.track_count, self.frame_count)
        track.add_detection(detection)
        self.tracks.append(track)

    def update_tracks(self, detections):
        self.frame_count += 1
        ndetections = len(detections)
        ntracks = len(self.tracks)
        print('update_tracks')
        print('ndetections = ' + str(ndetections))
        print('ntracks = ' + str(ntracks))
        # Matching between detections and tracks:
        if ndetections > 0 and ntracks > 0:
            print('matching')
            # Perform matching:
            track_ind, det_ind, unmatched_tracks, unmatched_detections = match_by_iou(self.tracks, detections, self.threshold_iou)
            # Add matched detections to tracks:
            for i in range(len(track_ind)):
                self.tracks[track_ind[i]].add_detection(detections[det_ind[i]])
            # Mark in the unmatched tracks that they are missing in this frame:
            for i in unmatched_tracks:
                self.tracks[i].mark_missing_frame()
            # Create new tracks for the unmatched detections:
            for i in unmatched_detections:
                self.create_track(detections[i])
        elif ndetections > 0:
            print('all new')
            # Create new tracks for all the detections:
            for i in range(ndetections):
                self.create_track(detections[i])
        elif ntracks > 0:
            print ('all missing')
            # Mark all tracks as missing in this frame:
            for i in range(ntracks):
                self.tracks[i].mark_missing_frame()

        # Remove old missing tracks:
        for i in range(ntracks - 1, -1, -1):
            if self.tracks[i].get_nframes_missing() > self.threshold_discard_track:
                del self.tracks[i]
        #self.tracks = list(filter(lambda tr: tr.get_nframes_missing() >= self.threshold_discard_track, self.tracks))

    def get_tracks(self):
        return self.tracks

    def clear(self):
        self.tracks = []
        self.track_count = 0
        self.frame_count = 0


def match_by_iou(tracks, detections, threshold_iou):
    print('match_by_iou')
    ndetections = len(detections)
    ntracks = len(tracks)
    # Compute IOU between existing tracks and new detections:
    max_size = max(ntracks, ndetections)
    iou_matrix = np.zeros((max_size, max_size)) # The positions that do not correspond to a real pair, stay with zeros.
    for i in range(ntracks):
        for j in range(ndetections):
            iou_matrix[i, j] = compute_iou(tracks[i].get_last_detection(), detections[j])
    # Apply the Hungarian method to the extended cost matrix (opposite of IOU matrix).
    track_ind, det_ind = linear_sum_assignment(-iou_matrix)
    track_ind = list(track_ind)
    det_ind = list(det_ind)
    # Loop over all the matches, to select those to discard:
    unmatched_tracks = []
    unmatched_detections = []
    for i in range(len(track_ind)):
        # Check the detection is assigned to a real track:
        if track_ind[i] < ntracks:
            # Check the track is assigned to a real detection:
            if det_ind[i] < ndetections:
                # Check IOU threshold:
                if iou_matrix[track_ind[i], det_ind[i]] < threshold_iou:
                    # Mark both track and detection as unmatched:
                    unmatched_tracks.append(track_ind[i])
                    unmatched_detections.append(det_ind[i])
            else:
                # Mark track as unmatched:
                unmatched_tracks.append(track_ind[i])
        else:
            # Mark detection as unmatched:
            unmatched_detections.append(det_ind[i])
    # Discard selected:
    for i in range(len(track_ind) - 1, -1, -1):
        if track_ind[i] in unmatched_tracks:
            del track_ind[i]
            del det_ind[i]
        elif det_ind[i] in unmatched_detections:
            del track_ind[i]
            del det_ind[i]
    return track_ind, det_ind, unmatched_tracks, unmatched_detections


def compute_iou(detection1, detection2):
    det1_xmax = detection1.x_center + 0.5 * detection1.width
    det2_xmax = detection2.x_center + 0.5 * detection2.width
    det1_xmin = detection1.x_center - 0.5 * detection1.width
    det2_xmin = detection2.x_center - 0.5 * detection2.width
    det1_ymax = detection1.y_center + 0.5 * detection1.height
    det2_ymax = detection2.y_center + 0.5 * detection2.height
    det1_ymin = detection1.y_center - 0.5 * detection1.height
    det2_ymin = detection2.y_center - 0.5 * detection2.height
    tb = min(det1_xmax, det2_xmax) - max(det1_xmin, det2_xmin)
    lr = min(det1_ymax, det2_ymax) - max(det1_ymin, det2_ymin)
    if tb < 0 or lr < 0 :
        intersection = 0
    else :
        intersection =  tb * lr
    union = detection1.width * detection1.height + detection2.width * detection2.height - intersection
    return intersection / union









