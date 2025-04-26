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


class Voice_Contrasting:
    '''
    1. Contrast the audio by inference
    2. Classify the audio by similarity
    '''
    def __init__(self,
        stdAudioSpeaker: dict,
        audioDirInput: str,
        modelPath: str = './Models/.pth',
        modelType: str = 'Ecapa-Tdnn',
        featureMethod: str = 'melspectrogram',
        decisionThreshold: float = 0.60,
        audioDuration: float = 4.20,
        outputRoot: str = "./",
        outputDirName: str = "",
        audioSpeakersDataName: str = "AudioSpeakerData"
    ):
        self.stdAudioSpeaker = stdAudioSpeaker
        self.audioDirInput = audioDirInput
        self.modelPath = modelPath
        self.modelDir = Path(modelPath).parent.__str__()
        self.modelName = Path(modelPath).stem.__str__()
        self.modelType = modelType
        self.featureMethod = featureMethod
        self.decisionThreshold = decisionThreshold
        self.audioDuration = audioDuration
        self.outputDir = Path(outputRoot).joinpath(outputDirName).as_posix()
        self.audioSpeakersDataPath = Path(self.outputDir).joinpath(audioSpeakersDataName).as_posix() + ".txt"

        os.makedirs(os.path.dirname(self.audioSpeakersDataPath), exist_ok = True)

    def getModel(self):
        '''
        Function to load model
        '''
        # Download Model
        if self.modelName in ['Ecapa-Tdnn_spectrogram', 'Ecapa-Tdnn_melspectrogram']:
            executeModelDownload(self.modelDir, self.modelName)

        # 获取模型
        DataSet = CustomDataset(data_list_path = None, feature_method = self.featureMethod)
        if self.modelType == 'Ecapa-Tdnn':
            self.Model = SpeakerIdetification(backbone = EcapaTdnn(input_size = DataSet.input_size))
        else:
            raise Exception(f'{self.modelType} 模型不存在！')

        # 指定使用设备
        self.Device = torch.device("cuda")

        # 加载模型
        self.Model.to(self.Device)
        Model_Dict = self.Model.state_dict()
        Param_State_Dir = torch.load(self.modelPath)
        for name, weight in Model_Dict.items():
            if name in Param_State_Dir.keys():
                if list(weight.shape) != list(Param_State_Dir[name].shape):
                    Param_State_Dir.pop(name, None)
        self.Model.load_state_dict(Param_State_Dir, strict = False)
        print(f"成功加载模型参数和优化方法参数：{self.modelPath}")
        self.Model.eval()

        #return self.Device, self.Model

    def inference(self):
        '''
        Function to infer 
        '''
        # 预测音频
        def infer(Audio_Path):
            data = load_audio(Audio_Path, mode = 'infer', feature_method = self.featureMethod, chunk_duration = self.audioDuration)
            data = data[np.newaxis, :]
            data = torch.tensor(data, dtype = torch.float32, device = self.Device)
            # 执行预测
            Feature = self.Model.backbone(data)
            return Feature.data.cpu().numpy()

        # 两两比对
        AudioSpeakersSim = {}
        for Speaker, Audio_Path_Std in self.stdAudioSpeaker.items():
            if os.path.exists(Audio_Path_Std):
                Feature1 = infer(Audio_Path_Std)[0]
            PatternList = []
            for Extension in ['*.flac', '*.wav', '*.mp3', '*.aac', '*.m4a', '*.wma', '*.aiff', '*.au', '*.ogg']:
                PatternList.extend(glob.glob(Path(self.audioDirInput).joinpath(Extension).as_posix()))
            for Audio_Path_Chk in PatternList:
                Feature2 = infer(Audio_Path_Chk)[0]
                # 对角余弦值
                Dist = np.dot(Feature1, Feature2) / (np.linalg.norm(Feature1) * np.linalg.norm(Feature2))
                if Dist > self.decisionThreshold:
                    print(f"{Audio_Path_Std} 和 {Audio_Path_Chk} 为同一个人，相似度为：{Dist}")
                else:
                    print(f"{Audio_Path_Std} 和 {Audio_Path_Chk} 不是同一个人，相似度为：{Dist}")
                if Audio_Path_Chk in AudioSpeakersSim.keys():
                    if float(Dist) <= float(AudioSpeakersSim[Audio_Path_Chk].split('|')[-1]):
                        continue
                AudioSpeakersSim[Audio_Path_Chk] = f"{Speaker if Dist > self.decisionThreshold else ''}|{Dist}"
        with open(self.audioSpeakersDataPath, mode = 'w', encoding = 'utf-8') as AudioSpeakersData:
            AudioSpeakersData.writelines([f"{Audio}|{SpeakerSim}\n" for Audio, SpeakerSim in AudioSpeakersSim.items()])