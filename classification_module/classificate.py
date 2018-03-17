from classification_module.prepare_ratio import get_model_ratio
from classification_module.numpy_proceed import preprocess_array
from doc2vec_module.train_model import get_model_for_genre
from doc2vec_module.constants import FileConstants
from gensim.models import Doc2Vec
from doc2vec_module import load
from pathlib import Path
from sklearn import svm
import numpy as np

#Пример работы для отдельной книги

lin_clf = svm.LinearSVC()
path_to_check = Path(__file__).parents[1].joinpath('books/test_book')
documents = load.get_doc(str(path_to_check))

print("Получаем модель для книги")
print("")
check_model = get_model_for_genre(documents)

print("Начинаем подгружать модели по жанрам")
print(" ")

prop_list = ["MODEL_ADVENTURE", "MODEL_ART", "MODEL_DETECTIVE", "MODEL_FANTASTIC", "MODEL_FANTASY", "MODEL_LOVE"]
genre_labels = ['приключения', 'искусство', 'детектив', 'фантастика', 'фэнтези', 'любовь']
train_list = []

for item in prop_list:
    print("Модель: {0}".format(item))
    model_prop = getattr(FileConstants, item)
    model = Doc2Vec.load(model_prop.fget(FileConstants()))
    train = preprocess_array(get_model_ratio(model))
    train_list.append(train)
    print(" ")

check_train = preprocess_array(np.array(check_model.docvecs[str(0)]))
np_train_list = np.asarray(train_list)
limit = 3
fix_ratio = 100
counter = 0
sum_temp = 0

for item in train_list:
    print("Расчёт для жанра \"{0}\"".format(genre_labels[counter]))
    lin_clf.fit(np_train_list, item[1:7])
    temp_ar_1 = lin_clf.predict(np.reshape(check_train, (1, -1)))
    sum_temp += np.sum(temp_ar_1)

    lin_clf.fit(np_train_list, item[8:14])
    temp_ar_2 = lin_clf.predict(np.reshape(check_train, (1, -1)))
    sum_temp += np.sum(temp_ar_2)

    lin_clf.fit(np_train_list, item[14:20])
    temp_ar_3 = lin_clf.predict(np.reshape(check_train, (1, -1)))
    sum_temp += np.sum(temp_ar_3)

    ratio = (sum_temp / 3) / 100
    print("Коэффициент для такого жанра, как {0}: {1}".format(genre_labels[counter], ratio))
    print(" ")
    counter +=1
    sum_temp = 0

