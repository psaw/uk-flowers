# Технический процесс пилота

GitHub `psaw/uk-flowers` имитирует исходный контур. SourceCraft `yandex-cloud-architects/yk-ttk-1` — среда проверки AI-коллег. Все Git-переносы выполняются с локального узла обмена.

## Первый запуск

```bash
git clone --origin onprem https://github.com/psaw/uk-flowers.git
cd uk-flowers
git remote add sourcecraft ssh://ssh.sourcecraft.dev/yandex-cloud-architects/yk-ttk-1.git
git config push.default nothing
# Если ключ не выбирается автоматически, укажите существующий зарегистрированный ключ:
git config core.sshCommand 'ssh -i /path/to/registered-key -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes'
```

SSH-ключ и корпоративный профиль должны соответствовать друг другу. Закрытый ключ не входит в репозиторий и не передаётся AI-коллеге.

## Проверки

```bash
bash scripts/pilot-check.sh smoke
bash scripts/pilot-check.sh quality
bash scripts/pilot-check.sh all
```

`smoke` выполняет автономные интеграционные тесты и сборку wheel. `quality` запускает линтер, форматирование, mypy, Bandit и весь pytest с исходным порогом покрытия 80%. `all` выполняет оба набора. Каждая проверка сохраняет свой журнал и код завершения; ошибки не прерывают сбор остальных результатов. Общий код ненулевой, если любая проверка не прошла.

Отчёты: `reports/pilot/summary-*.json`, JUnit XML, coverage XML, журналы и wheel. Генерируемые данные не коммитятся. Синтетические изображения и checkpoint создаются во временном каталоге pytest. Полное обучение, DVC/S3 и MLflow для smoke не нужны.

SourceCraft: `.sourcecraft/ci.yaml` запускает `all` на push в `main`, `poc/**`, `ai/**` и на PR в `main`. Конфигурация должна находиться в основной ветке репозитория. GitHub: новый workflow `Pilot smoke` запускает smoke после возврата ветки; существующий `Check` остаётся проверкой качества исходного проекта.

В SourceCraft используется native-кубик облачного воркера. Закреплённый uv устанавливается в `.cache/uv-bin`, Python — в `.cache/python`. Это позволяет выполнять проверки без подготовки пользовательского Docker-образа. Журналы и состояние CI читаются через `src run` и `src api`; браузер для автоматизации не требуется.

Фиксируется исходный `uv.lock`; GPU не требуется. На Linux исходный lock может устанавливать CUDA-библиотеки как зависимости стандартного PyTorch. Выделенный CPU-профиль зависимостей — отдельное улучшение, текущий lock не меняется.

## Обмен ветками

```bash
# Опубликовать локальную принятую ветку в SourceCraft:
bash scripts/pilot-transfer.sh publish poc/workflow-bootstrap main

# Получить актуальное состояние обеих сторон:
bash scripts/pilot-transfer.sh status

# Вернуть принятую ветку SourceCraft отдельной веткой в GitHub:
bash scripts/pilot-transfer.sh return main sourcecraft/workflow-bootstrap
```

Скрипт проверяет чистоту клона, происхождение от актуального `onprem/main` и SHA после push. Он не выполняет force-push, слияние или автоматическое принятие PR. Возврат разрешён только в ветки `sourcecraft/*`.

Если исходная `main` обновилась, создайте рабочую ветку от нужной ветки SourceCraft, выполните `git merge onprem/main`, разрешите конфликты и повторите проверки. Затем опубликуйте эту ветку. Приёмка возврата выполняется PR в GitHub с повторными проверками. Отмена принятого изменения — отдельный revert-коммит и PR.

Повторная передача того же SHA не создаёт новых коммитов. Автоматического двустороннего зеркалирования и webhook-петли нет.

## Первая задача AI-коллеге

Используйте постановку `pilot/tasks/001-cli-tests.md` и инструкции `AGENTS.md`. Для запуска, просмотра AI-задач и статусов CI необходим авторизованный интерфейс, SourceCraft CLI или REST API. Git-доступ по SSH сам по себе не предоставляет доступ к этим операциям.

## Границы проверки

Этот процесс проверяет Git-обмен и разработку на учебном ML-проекте. Работа в реальных Gitea/Jenkins, сетевые ACL ТТК, бизнес-эффект и качество обученной модели требуют отдельных проверок.
