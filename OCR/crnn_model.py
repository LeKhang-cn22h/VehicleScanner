import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, num_classes, input_size=1024):  # thêm input_size ở đây
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # giữ nguyên chiều rộng
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1), (2, 1))
        )

        self.rnn = nn.LSTM(
            input_size=input_size,  # giờ input_size đã hợp lệ
            hidden_size=256,
            num_layers=2,
            bidirectional=True
        )
        self.fc = nn.Linear(512, num_classes)
        # self.lstm=nn.LSTM(input_size, 256, bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, x):
        # self.lstm.flatten_parameters()

        B, C, H, W = x.size()

        x = self.cnn(x)  # -> [B, C, H', W']
        B, C, H, W = x.size()

        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.reshape(B, W, C * H)  # [B, W, C*H]
        print("Trước khi view:", x.shape)
        x = x.permute(1, 0, 2)  # [W, B, C*H]  => chuẩn input cho LSTM

        # self.rnn.flatten_parameters()
        # if x.is_cuda:
        #     pass  # Không cần gọi flatten_parameters() trên GPU
        # else:
        #     self.lstm.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.fc(x)  # [W, B, num_classes]
        # recurrent, _ = self.lstm(x)

        return x
