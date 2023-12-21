# Оценка комфортности места для выполнения домашнего задания по следующим парамтетрам: уровень шума, удобство рабочего
# места, возможность не отвлекаться
# Идеальное место для выполнения домашки должно быть тихим (абсолютная тишина - 1), с подходящими столом и стулом
# (рабочий стол подходящей высоты и достаточной ширины, не слишком расслабляющий, но при этом удобный стул - 1) и с
# минимумом отвлекающих моментов (чем меньше забирающих внимание факторов - тем больше возможность на них не
# реагировать), таких как, например, люди, телевизор или домашние животные (полное отсутствие мешающих сосредоточиться
# факторов - 1)
# Оценивать будем библиотеку, кафе, университет, квартиру и коворкинг


import torch

library = torch.tensor([[0.9, 0.8, 0.9]])  # Заинтриговавшая книга на полке может очень привлекать
cafe = torch.tensor([[0.3, 0.4, 0.3]])
university = torch.tensor([[0.6, 0.8, 0.7]])  # На переменах в аудиторию может доноситься шум, а сидящий по соседству друг так и наровоит поболтать
apartment = torch.tensor([[0.7, 0.7, 0.5]])  # У меня стул без спинки, кот и собака
coworking = torch.tensor([[0.8, 0.9, 0.8]])

dataset = [
    (library, torch.tensor([[1.0]])),
    (cafe, torch.tensor([[0.3]])),
    (university, torch.tensor([[0.7]])),
    (apartment, torch.tensor([[0.6]])),
    (coworking, torch.tensor([[0.8]])),
]

torch.manual_seed(1750)

weights = torch.rand((1, 3), requires_grad=True)  # В матрице 1 строка и 3 колонки
bias = torch.rand((1, 1), requires_grad=True)

mse_loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([weights, bias], lr=1e-3)


def predict_comfort_score(obj: torch.Tensor) -> torch.Tensor:
    return obj @ weights.T + bias


def calc_loss(predicted_value: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    return mse_loss_fn(predicted_value, ground_truth)


num_epochs = 10

for i in range(num_epochs):
    for x, y in dataset:
        optimizer.zero_grad()
        threat_score = predict_comfort_score(x)

        loss = calc_loss(threat_score, y)
        loss.backward()
        print(loss)
        optimizer.step()
