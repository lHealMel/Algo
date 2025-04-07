from models import MNet
import torch.nn as nn
from dataset import *
from torch.utils.data import DataLoader


def train(batch_size=4, max_epoch=1000, threshold=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    train_dataset = TrainDataset()
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TestDataset()
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(max_epoch):
        model.train()

        for X, Y in train_data_loader:
            X = X.to(device)
            Y = Y.to(device)

            outputs = model(X)
            loss = criterion(outputs, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for x, y in test_data_loader:
                x = x.to(device)
                y = y.to(device)

                # 14
                outputs = model(x)
                loss = criterion(outputs, y)

                # 15
                test_loss += loss.item()

                # 🔥 정확도 계산: 오차가 threshold 이하인 경우 정답 처리
                correct += (torch.abs(outputs - y) < threshold).sum().item()
                print("output: ", outputs, "\nreal value:", y, "\n")
        # 16
        total_elements = len(test_dataset) * y.shape[1]  # 전체 요소 개수
        accuracy = correct / total_elements  # 정확도를 전체 요소 개수로 나눔
        print(f"epoch {epoch + 1} - test loss: {test_loss / len(test_data_loader):.4f}, accuracy: {accuracy:.4f}")

    print('Learning finished')


if __name__ == '__main__':
    train()
