import torch
import os
import sys
import glob
import numpy as np

from pathlib import Path
current_dir = Path(__file__).absolute().parent.as_posix()
sys.path.insert(0, f"{current_dir}")
os.chdir(current_dir)

from modules.ECAPA_TDNN import EcapaTdnn, SpeakerIdetification
from data_utils.reader import load_audio, CustomDataset
from utils.downloader import executeModelDownload


class Voice_Identifying:
    '''
    1. Contrast the audio by inference
    2. Classify the audio by similarity
    '''
    def __init__(self,
        StdAudioSpeaker: dict,
        Audio_Dir_Input: str,
        Model_Path: str = './Models/.pth',
        Model_Type: str = 'Ecapa-Tdnn',
        Feature_Method: str = 'melspectrogram',
        DecisionThreshold: float = 0.60,
        Duration_of_Audio: float = 4.20,
        Output_Root: str = "./",
        Output_DirName: str = "",
        AudioSpeakersData_Name: str = "AudioSpeakerData"
    ):
        self.StdAudioSpeaker = StdAudioSpeaker
        self.Audio_Dir_Input = Audio_Dir_Input
        self.Model_Path = Model_Path
        self.Model_Dir = Path(Model_Path).parent.__str__()
        self.Model_Name = Path(Model_Path).stem.__str__()
        self.Model_Type = Model_Type
        self.Feature_Method = Feature_Method
        self.DecisionThreshold = DecisionThreshold
        self.Duration_of_Audio = Duration_of_Audio
        self.Output_Dir = Path(Output_Root).joinpath(Output_DirName).as_posix()
        self.AudioSpeakersData_Path = Path(self.Output_Dir).joinpath(AudioSpeakersData_Name).as_posix() + ".txt"

        os.makedirs(os.path.dirname(self.AudioSpeakersData_Path), exist_ok = True)

    def getModel(self):
        '''
        Function to load model
        '''
        # Download Model
        if self.Model_Name in ['Ecapa-Tdnn_spectrogram', 'Ecapa-Tdnn_melspectrogram']:
            executeModelDownload(self.Model_Dir, self.Model_Name)

        # 获取模型
        DataSet = CustomDataset(data_list_path = None, feature_method = self.Feature_Method)
        if self.Model_Type == 'Ecapa-Tdnn':
            self.Model = SpeakerIdetification(backbone = EcapaTdnn(input_size = DataSet.input_size))
        else:
            raise Exception(f'{self.Model_Type} 模型不存在！')

        # 指定使用设备
        self.Device = torch.device("cuda")

        # 加载模型
        self.Model.to(self.Device)
        Model_Dict = self.Model.state_dict()
        Param_State_Dir = torch.load(self.Model_Path)
        for name, weight in Model_Dict.items():
            if name in Param_State_Dir.keys():
                if list(weight.shape) != list(Param_State_Dir[name].shape):
                    Param_State_Dir.pop(name, None)
        self.Model.load_state_dict(Param_State_Dir, strict = False)
        print(f"成功加载模型参数和优化方法参数：{self.Model_Path}")
        self.Model.eval()

        #return self.Device, self.Model

    def inference(self):
        '''
        Function to infer 
        '''
        # 预测音频
        def infer(Audio_Path):
            data = load_audio(Audio_Path, mode = 'infer', feature_method = self.Feature_Method, chunk_duration = self.Duration_of_Audio)
            data = data[np.newaxis, :]
            data = torch.tensor(data, dtype = torch.float32, device = self.Device)
            # 执行预测
            Feature = self.Model.backbone(data)
            return Feature.data.cpu().numpy()

        # 两两比对
        AudioSpeakersSim = {}
        for Speaker, Audio_Path_Std in self.StdAudioSpeaker.items():
            if os.path.exists(Audio_Path_Std):
                Feature1 = infer(Audio_Path_Std)[0]
            PatternList = []
            for Extension in ['*.flac', '*.wav', '*.mp3', '*.aac', '*.m4a', '*.wma', '*.aiff', '*.au', '*.ogg']:
                PatternList.extend(glob.glob(Path(self.Audio_Dir_Input).joinpath(Extension).as_posix()))
            for Audio_Path_Chk in PatternList:
                Feature2 = infer(Audio_Path_Chk)[0]
                # 对角余弦值
                Dist = np.dot(Feature1, Feature2) / (np.linalg.norm(Feature1) * np.linalg.norm(Feature2))
                if Dist > self.DecisionThreshold:
                    print(f"{Audio_Path_Std} 和 {Audio_Path_Chk} 为同一个人，相似度为：{Dist}")
                else:
                    print(f"{Audio_Path_Std} 和 {Audio_Path_Chk} 不是同一个人，相似度为：{Dist}")
                if Audio_Path_Chk in AudioSpeakersSim.keys():
                    if float(Dist) <= float(AudioSpeakersSim[Audio_Path_Chk].split('|')[-1]):
                        continue
                AudioSpeakersSim[Audio_Path_Chk] = f"{Speaker if Dist > self.DecisionThreshold else ''}|{Dist}"
        with open(self.AudioSpeakersData_Path, mode = 'w', encoding = 'utf-8') as AudioSpeakersData:
            AudioSpeakersData.writelines([f"{Audio}|{SpeakerSim}\n" for Audio, SpeakerSim in AudioSpeakersSim.items()])