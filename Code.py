import cv2
import mediapipe as mp

mp_h = mp.solutions.hands
hands = mp_h.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def detect_gst(landmarks):
    t_tip = landmarks.landmark[mp_h.HandLandmark.THUMB_TIP]
    t_ip = landmarks.landmark[mp_h.HandLandmark.THUMB_IP]
    t_mcp = landmarks.landmark[mp_h.HandLandmark.THUMB_MCP]
    
    i_tip = landmarks.landmark[mp_h.HandLandmark.INDEX_FINGER_TIP]
    i_pip = landmarks.landmark[mp_h.HandLandmark.INDEX_FINGER_PIP]
    i_dip = landmarks.landmark[mp_h.HandLandmark.INDEX_FINGER_DIP]
    
    m_tip = landmarks.landmark[mp_h.HandLandmark.MIDDLE_FINGER_TIP]
    m_pip = landmarks.landmark[mp_h.HandLandmark.MIDDLE_FINGER_PIP]
    m_dip = landmarks.landmark[mp_h.HandLandmark.MIDDLE_FINGER_DIP]
    
    r_tip = landmarks.landmark[mp_h.HandLandmark.RING_FINGER_TIP]
    r_pip = landmarks.landmark[mp_h.HandLandmark.RING_FINGER_PIP]
    r_dip = landmarks.landmark[mp_h.HandLandmark.RING_FINGER_DIP]
    
    p_tip = landmarks.landmark[mp_h.HandLandmark.PINKY_TIP]
    p_pip = landmarks.landmark[mp_h.HandLandmark.PINKY_PIP]
    p_dip = landmarks.landmark[mp_h.HandLandmark.PINKY_DIP]
    
    if (t_tip.y < t_ip.y and t_tip.y < t_mcp.y and
        i_tip.y > i_dip.y and m_tip.y > m_dip.y and
        r_tip.y > r_dip.y and p_tip.y > p_dip.y):
        return "OK"
    
    if (t_tip.y > t_ip.y and t_tip.y > t_mcp.y and
        i_tip.y > i_dip.y and m_tip.y > m_dip.y and
        r_tip.y > r_dip.y and p_tip.y > p_dip.y):
        return "Not OK"
    
    if (t_tip.y > t_ip.y and
        i_tip.y > i_dip.y and m_tip.y > m_dip.y and
        r_tip.y > r_dip.y and p_tip.y > p_dip.y):
        return "Yes"
    
    if (t_tip.y < t_mcp.y and
        i_tip.y < i_pip.y and m_tip.y < m_pip.y and
        r_tip.y < r_pip.y and p_tip.y < p_pip.y):
        return "Hello"
    
    if (i_tip.y < i_pip.y and m_tip.y < m_pip.y and
        r_tip.y > r_dip.y and p_tip.y > p_dip.y):
        return "Peace"
    
    if (i_tip.y < i_pip.y and m_tip.y > m_dip.y and
        r_tip.y > r_dip.y and p_tip.y > p_dip.y and
        t_tip.y < t_mcp.y):
        return "No"
    
    if (i_tip.y < i_pip.y and m_tip.y > m_dip.y and
        r_tip.y > r_dip.y and p_tip.y < p_dip.y and
        t_tip.y < t_ip.y):
        return "I Love You"
    
    if (t_tip.y > t_ip.y and i_tip.y > i_dip.y and
        m_tip.y > m_dip.y and r_tip.y > r_dip.y and
        p_tip.y < p_dip.y):
        return "I Hate You"
    
    return "Undefined Gesture"

while True:
    ret, frm = cap.read()
    if not ret:
        break

    rgb_frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb_frm)

    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            draw.draw_landmarks(frm, landmarks, mp_h.HAND_CONNECTIONS)
            gst = detect_gst(landmarks)
            if gst:
                cv2.putText(frm, gst, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frm)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
