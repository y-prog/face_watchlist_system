import os,time,json,cv2,streamlit as st,numpy as np,pandas as pd
from PIL import Image
from oop_face_detector import FaceDetector

st.set_page_config(layout="wide")
st.title("Face Watchlist System")

video_file=st.file_uploader("Upload Video",type=["mp4"])
known_face_files=st.file_uploader("Upload Known Faces",type=["jpg","jpeg","png"],accept_multiple_files=True)

frame_interval=st.number_input("Frame Interval (seconds)",min_value=1,value=2)
threshold_similarity=st.slider("Similarity Threshold",0.0,1.0,0.45)

frame_dir="output/saved_frames"
json_path="output/detections.json"

def clear_dir(path):
    os.makedirs(path,exist_ok=True)
    for f in os.listdir(path):
        os.remove(os.path.join(path,f))

def save_frames(faces_in_frames):
    clear_dir(frame_dir)
    rows=[]
    for i,(t,frame,dets) in enumerate(faces_in_frames):
        cv2.imwrite(f"{frame_dir}/frame_{i:04d}.jpg",frame)
        for name,score,bbox in dets:
            rows.append({
                "name":name,
                "score":float(score),
                "bbox":str([int(b) for b in bbox]),
                "secs":i*frame_interval
            })
    os.makedirs("output",exist_ok=True)
    with open(json_path,"w") as f:
        json.dump(rows,f,indent=2)
    return rows

# =========================
# PROCESSING
# =========================

if st.button("Start Processing") and video_file and known_face_files:

    os.makedirs("uploads/video",exist_ok=True)
    os.makedirs("uploads/known_faces",exist_ok=True)
    os.makedirs("output",exist_ok=True)
    os.makedirs(frame_dir,exist_ok=True)

    video_path="uploads/video/input.mp4"

    with open(video_path,"wb") as f:
        f.write(video_file.read())

    for f in os.listdir("uploads/known_faces"):
        os.remove(os.path.join("uploads/known_faces",f))

    for file in known_face_files:
        with open(os.path.join("uploads/known_faces",file.name),"wb") as f:
            f.write(file.read())

    progress=st.progress(0)

    fdec=FaceDetector(frame_interval=frame_interval,threshold_similarity=threshold_similarity)

    known_embeddings=fdec.load_known_embeddings("uploads/known_faces")
    progress.progress(25)

    frames=fdec.extract_frames_every_n_seconds(video_path)
    progress.progress(50)

    faces_in_frames=fdec.identify_faces_in_frames(frames,known_embeddings)
    progress.progress(75)

    rows=save_frames(faces_in_frames)
    progress.progress(100)

    st.success("Processing Complete")

# =========================
# TABLE
# =========================

st.subheader("Detections Table")

if os.path.exists(json_path):

    with open(json_path,"r") as f:
        rows=json.load(f)

    df=pd.DataFrame(rows)

    st.dataframe(df,use_container_width=True)

    people=["All"]+sorted(df["name"].unique())
    selected=st.selectbox("Filter by Person",people)

    if selected!="All":
        st.dataframe(df[df["name"]==selected],use_container_width=True)

else:
    st.info("No detections yet")

# =========================
# VIDEO PLAYER
# =========================

st.subheader("Frame Player")

files=sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")]) if os.path.exists(frame_dir) else []

if files:

    max_frame=len(files)-1

    frame_idx=st.slider("Timeline (Frame Index)",0,max_frame,0)

    speed=st.slider("Playback Speed (FPS)",1,10,2)

    video_width=st.slider("Video Width",300,1400,700)

    col1,col2=st.columns(2)

    play=col1.button("▶ Play")
    stop=col2.button("⏹ Stop")

    placeholder=st.empty()

    if play:

        i=frame_idx

        while i<len(files):

            if stop:
                break

            img=Image.open(os.path.join(frame_dir,files[i]))

            placeholder.image(img,width=video_width)

            time.sleep(1/speed)

            i+=1

    else:

        img=Image.open(os.path.join(frame_dir,files[frame_idx]))

        placeholder.image(img,width=video_width)