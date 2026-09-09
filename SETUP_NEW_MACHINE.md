# Развёртывание на новой машине (офлайн после этого шага)

## 1. Перенос кода

Проще всего — `git clone` из репозитория (код + результаты + датасеты DUD-E,
всё, что не в `.gitignore`):

```
git clone https://github.com/adilhanzheksengaliev2006-alt/onkohuy.git
```

## 2. Перенос того, что НЕ в git (флешкой, эти папки в .gitignore специально -
слишком тяжёлые/бинарные для git)

Скопировать с этой машины на новую, в ТЕ ЖЕ относительные пути внутри проекта:

| Что | Откуда | Размер | Зачем |
|---|---|---|---|
| `tools/vina.exe` | `tools/` | 1.2 МБ | сам движок докинга |
| `DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt` | `DiffSBDD/checkpoints/` | 17.8 МБ | веса генеративной модели (Test B) |
| `structures/` (целиком) | `structures/` | ~22 МБ | скачанные PDB + подготовленные `_receptor.pdbqt` - без них придётся перескачивать/переготавливать через сеть |
| `runs/_ligand_smiles_cache.json` | `runs/` | небольшой | **критично для офлайна** - кэш всех уже полученных SMILES, без него часть Test 5/6/15 полезет в сеть |
| `runs/test_a_*/ligands.json` и `results.jsonl` | `runs/` | по мишени | уже собранные датасеты + частично сделанный докинг (resume продолжит отсюда) |
| `module_generative/` (если нужен) | | 318 МБ | отдельная ветка работы, не часть 18-тестового протокола - переносить, только если это тоже нужно |

## 3. Python-окружения

Два раздельных conda-окружения (несовместимые версии зависимостей):

```
conda create -n molgen python=3.10.20
conda activate molgen
pip install -r environment_molgen_requirements.txt

conda create -n diffsbdd python=3.10.4
conda activate diffsbdd
pip install -r environment_diffsbdd_requirements.txt
```

Оба requirements-файла - в корне проекта, экспортированы `pip freeze` с этой
машины 09.09.2026.

## 4. WSL2 (gnina CNN-рескоринг + fpocket)

Требуется WSL2 Ubuntu с установленными:
- **gnina** (CUDA-сборка, использует GPU через WSL passthrough - на RTX 4080
  должно быть быстрее, чем на GTX 1650 Ti, где это разрабатывалось)
- **fpocket** (conda-forge пакет)

Точные команды установки этой сессией не документировались построчно (делалось
в более ранней сессии) - при необходимости переустановить с нуля: gnina
собирается из исходников с CUDA-поддержкой (см. официальный репозиторий
gnina/gnina), fpocket ставится через `conda install -c conda-forge fpocket`
внутри WSL-окружения.

**Важно**: `.wslconfig` на этой машине настроен на `memory=4GB` (было 3GB,
поднято из-за нехватки под несколько параллельных gnina-процессов) - на
машине со 120ГБ RAM можно поставить значительно больше, это уберёт целый
класс потенциальных сбоев.

## 5. Проверка готовности к офлайну

После переноса, ДО отключения интернета:

```
python check_offline_readiness.py
```

Должно показать `76/98` (или больше, если Task 1/онбординг успели продвинуться
дальше на исходной машине перед переносом) мишеней с `fully_offline_ready: True`.
Для НЕ готовых - см. вывод скрипта, что именно отсутствует (структура/рецептор/
датасет/SMILES).

## 6. Настройки энергосбережения

На этой машине это уже отключено (`powercfg /change standby-timeout-* 0`,
аналогично hibernate) - при многочасовом/многодневном докинге сон/гибернация
обрывает все фоновые процессы без предупреждения (обнаружено в бою). На новой
машине сделать то же самое ДО запуска длинного докинга:

```
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
```

## 7. Запуск

```
python run_docking_queue.py --only-gated
```

Resume-safe - можно останавливать/перезапускать в любой момент без потери
прогресса (`phase_dock()` пропускает уже сделанные лиганды по `results.jsonl`).
