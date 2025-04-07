import torch

def cr_data(n_samples=100, n_features=8, type=None):
    if type is None:
        print('specify the type : {train, test}')
        return -1
    # 설정
    torch.manual_seed(234)

    # 첫 번째 벡터: 0 또는 1을 가지는 이진 벡터
    binary_vector = torch.randint(0, 2, (n_samples, n_features)).float()

    # 두 번째 벡터: 첫 번째 벡터를 기반으로 0~1 사이의 실수값 생성
    real_valued_vector = torch.where(
        binary_vector == 1,
        torch.rand(n_samples, n_features) * 0.5 + 0.5,  # 0.5 ~ 1.0 (1일 때)
        torch.rand(n_samples, n_features) * 0.5  # 0.0 ~ 0.5 (0일 때)
    )
    if (type == 'train'):
        return binary_vector[:80], real_valued_vector[:80]
    elif (type == 'test'):
        return binary_vector[80:], real_valued_vector[80:]
    else:
        print('Error')
        return -1


if __name__ == '__main__':
    cr_data()