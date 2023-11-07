import streamlit as st
import torch
import torchvision
from model import Resnet
from PIL import Image
import matplotlib.pyplot as plt

def predict(image, labels, model):
            
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor() 
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    model.eval()
    outputs = model(image)

    y_prob = torch.nn.functional.softmax(outputs.squeeze(0), dim=-1)
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)

    results = []
    for prob, idx in zip(sorted_prob, sorted_indices):
        results.append((labels[idx.item()], prob.item()))

    return results
    

def main():
    model = Resnet(num_classes=10)
    model.load_state_dict(torch.load("model.pth", map_location='cpu'))

    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    st.sidebar.title("衣服の画像認識アプリ")
    st.sidebar.write("画像認識モデルを使って衣服の種類を判定します。")

    st.sidebar.write("")

    img_source = st.sidebar.radio("画像のソースを選択してください",
                                  ("画像をアップロード", "画像を撮影"))
    
    if img_source == "画像をアップロード":
        img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
    else:
        img_file = st.camera_input("カメラで撮影")

    if img_file is not None:
        with st.spinner("計算中・・・"):
            img = Image.open(img_file)
            st.image(img, caption="対象画像")
            st.write("")

            results = predict(img, labels, model)

            st.subheader("判定結果")
            num_top = 5
            for result in results[:num_top]:
                st.write(str(round(result[1] * 100, 2)) + "%の確率で" + result[0] + "です。")

            pie_labels = [result[0] for result in results[:num_top]]
            pie_labels.appned("Other")
            pie_probs = [result[1] for result in results[:num_top]]
            pie_probs.appned(sum([result[1] for result in results]))

            fig, ax = plt.subplots()
            wedgeprops = {"width":0.3, "edgecolor":"white"}
            textprops = {"fontsize":6}
            ax.pie(pie_probs, labels=pie_labels, counterclocke=False, startangle=90,
                   textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)
            st.pyplot(fig)


    st.sidebar.write("")
    st.sidebar.write("")

    st.sidebar.caption('"このアプリは[FashionMnist]を訓練データとして扱っています \n \
                       Copyright (c) 2017 Zalando SE \n \
                       Released under the MIT license \n \
                       https://github.com/zalandoresearch/fashion-mnist#license"')



if __name__ == "__main__":
    main()
