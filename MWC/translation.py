import bpy

# Standard translation dictionary structure for Blender's native i18n
translations_dict = {
    "ru_RU": {
        # Context "*" matches any context if there is no context-specific translation
        ("*", "Metaball Weight Container (MWC) 1.0"): "Metaball Weight Container (MWC) 1.0",
        
        # Panel headers
        ("*", "1. Generation & Cache Creation"): "1. Генерация и создание кэша",
        ("*", "2. Viewport & Editing"): "2. Отображение и редактирование",
        ("*", "3. Weight Transfer & Baking"): "3. Перенос весов и запекание",
        
        # Operators
        ("*", "Clear Cache"): "Очистить кэш",
        ("*", "Spawn Viewport Metaballs"): "Показать во вьюпорте",
        ("*", "Clear Viewport Metaballs"): "Убрать из вьюпорта",
        ("*", "Create Metaballs"): "Создать метаболы",
        ("*", "Apply Weights"): "Применить веса",
        ("*", "Initialize Custom Curve"): "Инициализировать кастомную кривую",
        ("*", "Select Nearest Metaball"): "Выбрать ближайший к 3D курсору",
        ("*", "Add Metaball"): "Добавить метабол",
        ("*", "Snap Active to Cursor"): "Привязать к 3D курсору",
        ("*", "Add Bone Weight"): "Добавить вес кости",
        ("*", "Remove Bone Weight"): "Удалить вес кости",
        ("*", "Save Metaballs to Cache"): "Сохранить в кэш",
        
        # UI labels
        ("*", "Source Data (Creation):"): "Исходные данные (Создание):",
        ("*", "Source Mesh"): "Исходный меш",
        ("*", "Creation Parameters:"): "Параметры создания:",
        ("*", "Alpha"): "Alpha (Масштаб)",
        ("*", "Alpha (Scale)"): "Alpha (Масштаб)",
        ("*", "Subdivision K"): "Коэф. дробления (K)",
        ("*", "Subdivision Coeff (K)"): "Коэф. дробления (K)",
        ("*", "Merge Close"): "Сливать близкие",
        ("*", "Merge Factor"): "Коэф. слияния",
        ("*", "Creation Type"): "Grouping",
        ("*", "Grouping"): "Группировка",
        ("*", "Symmetry"): "Симметрия",
        ("*", "Create"): "Создать",
        ("*", "Viewport Preview"): "Предпросмотр во вьюпорте",
        ("*", "Show Preview"): "Показать предпросмотр",
        ("*", "Select Nearest"): "Выбрать ближайший",
        ("*", "Color by Active Bone"): "Окрасить по активной кости",
        ("*", "Active Metaball Editor"): "Редактор активного метабола",
        ("*", "Snap to Cursor"): "Привязать к 3D курсору",
        ("*", "Bone Weights"): "Веса костей",
        ("*", "Save to Cache"): "Сохранить в кэш",
        ("*", "Weight Application:"): "Применение весов:",
        ("*", "Target Mesh"): "Целевой меш",
        ("*", "Transfer Parameters:"): "Параметры переноса:",
        ("*", "Geodesic Distance"): "По ребрам (Geodesic)",
        ("*", "Geodesic Performance Mode"): "Режим геодезики",
        ("*", "Custom Falloff Curve"): "Кастомная кривая спада",
        ("*", "Wyvill Exponent (n)"): "Показатель Wyvill (n)",
        ("*", "Mixing Exponent (q)"): "Показатель смешивания (q)",
        ("*", "Threshold (tau)"): "Порог (tau)",
        ("*", "R Falloff Coeff"): "Коэф. спада R",
        ("*", "Normal Filter"): "Фильтр нормалей",
        ("*", "Strictness (p)"): "Строгость (p)",
        ("*", "Smoothing"): "Сглаживание",
        ("*", "Smoothing Strength"): "Сила сглаживания",
        ("*", "Iterations"): "Итерации",
        ("*", "Apply"): "Применить",
        ("*", "Metaballs Cache Status:"): "Статус кэша метаболов:",
        ("*", "Empty (Run 'Create')"): "Пусто (Запустите «Создать»)",
        ("*", "Select a metaball in the viewport to edit it"): "Выберите метабол во вьюпорте для его редактирования",
        
        # Enums
        ("*", "Single Object"): "Один объект",
        ("*", "Multiple Objects"): "Несколько объектов",
        ("*", "Sequential"): "Последовательный (SEQ)",
        ("*", "Thread Pool"): "Пул потоков (THREAD)",
        
        # Warning boxes / dialogs
        ("*", "Warning: Object contains self-intersecting geometry!"): "Внимание: Объект содержит самопересечения геометрии!",
        ("*", "This can lead to weight distribution instability."): "Это может привести к нестабильности распределения весов.",
        ("*", "Do you really want to continue?"): "Вы действительно хотите продолжить?",
        
        # Property descriptions / Tooltips
        ("*", "Metaball radius coefficient"): "Коэффициент радиуса метаболов",
        ("*", "Exponent n in Wyvill formula"): "Степень n в функции Wyvill",
        ("*", "Transition mixing stiffness parameter"): "Параметр жесткости перехода смешивания",
        ("*", "Threshold for cutting off small weights"): "Порог отсечения малых весов",
        ("*", "R_falloff radius coefficient relative to average radius"): "Коэффициент радиуса спада R_falloff относительно среднего радиуса",
        ("*", "Coefficient for local subdivision of long edges (L > K * max(R1, R2))"): "Коэффициент для локального дробления длинных ребер (L > K * max(R1, R2))",
        ("*", "Merge close metaballs with same dominant bone"): "Объединять близкие метаболы с одинаковой преобладающей костью",
        ("*", "Merge distance coefficient (factor * (R1 + R2))"): "Коэффициент расстояния слияния (factor * (R1 + R2))",
        ("*", "Whether to create a single group of metaballs or group by geometric islands"): "Создавать ли единую группу метаболов или группировать по геометрическим островам",
        ("*", "All metaballs belong to one group/family"): "Все метаболлы принадлежат одной группе/семейству",
        ("*", "Separate metaballs into families based on geometry islands"): "Разделить метаболлы по семействам на основе островов геометрии",
        ("*", "Generate only left half of metaballs and automatically mirror to the right"): "Генерировать только левую половину метаболов и автоматически отзеркаливать на правую",
        ("*", "Target mesh onto which weights from metaballs will be projected"): "Целевой меш, на который будут спроецированы веса из метаболов",
        ("*", "Enable weight filtering by vertex normals to prevent leaks. Do not enable if the mesh has thickness (solidify)."): "Включить фильтрацию весов по нормалям вершин для предотвращения протекания. Не включайте, если меш имеет толщину.",
        ("*", "Strictness exponent of the normal filter (higher p = stricter filtering)"): "Показатель степени строгости фильтра нормалей (выше p - строже фильтрация)",
        ("*", "Smooth transferred weights across adjacent vertices"): "Сглаживать перенесенные веса по соседним вершинам",
        ("*", "Weight smoothing strength per iteration"): "Сила сглаживания весов на каждой набегающей итерации",
        ("*", "Number of passes (iterations) of smoothing"): "Количество проходов (итераций) сглаживания",
        ("*", "Use geodesic distance along mesh edges (Dijkstra) instead of Euclidean to prevent weight leakage"): "Использовать геодезическое расстояние по ребрам сетки (алгоритм Дейкстры) вместо Евклидова для предотвращения протекания весов",
        ("*", "Choose multi-threading/multi-processing method for Dijkstra calculation"): "Выберите метод распараллеливания (потоки/процессы) для расчета Дейкстры",
        ("*", "Adapt metaball radius in joints and center of bones using armature joint positions"): "Адаптировать радиус метаболов в суставах и центрах костей с помощью позиций суставов скелета",
        ("*", "Skeletal Armature object to retrieve joint positions from"): "Объект скелета (Armature) для считывания позиций суставов",
        ("*", "Radius multiplier near joint connections"): "Множитель радиуса в местах соединения костей (суставах)",
        ("*", "Radius multiplier in the middle of long bones"): "Множитель радиуса на середине длинных костей",
        ("*", "Limit metaball radius to local mesh thickness via raycast to prevent leaks"): "Ограничивать радиус метабола локальной толщиной исходного меша (через лучи) для предотвращения протекания весов",
        ("*", "Clamping factor for thickness (radius = min(radius, thickness * factor))"): "Ограничивающий коэффициент толщины (радиус = min(радиус, толщина * коэф.))",
        ("*", "Use custom Bezier curve editor for weight falloff instead of Wyvill formula"): "Использовать интерактивный редактор кривых Безье вместо фиксированной формулы Wyvill",
        
        # Operators descriptions
        ("*", "Delete the saved metaballs cache file"): "Удалить сохраненный файл кэша метаболов",
        ("*", "Spawn metaballs from cache as actual scene objects for viewport editing"): "Отобразить метаболлы из кэша в виде объектов сцены во вьюпорте для редактирования",
        ("*", "Remove metaball objects from the scene viewport without deleting the cache file"): "Убрать объекты метаболлов из вьюпорта, оставив файл кэша нетронутым",
        ("*", "Generate metaballs from vertex weights of the source mesh and save to cache"): "Генерировать метаболлы на основе весов вершин исходного меша и сохранить в кэш",
        ("*", "Bake weights from cached metaballs onto the target mesh"): "Запечь веса из кэшированных метаболов на целевой меш",
        ("*", "Initialize the custom curve node group safely in write-allowed context"): "Инициализировать группу узлов кастомной кривой безопасно",
        ("*", "Select the metaball closest to the 3D Cursor"): "Выбрать ближайший к 3D курсору",
        ("*", "Create a new metaball at the 3D Cursor location"): "Добавить новый метабол в позиции 3D курсора",
        ("*", "Add a new bone weight custom property to the active metaball"): "Добавить новое пользовательское свойство веса кости для активного метабола",
        ("*", "Remove this bone weight custom property"): "Удалить это пользовательское свойство веса кости",
        ("*", "Move active metaball to the 3D Cursor location"): "Привязать активный метабол к 3D курсору",
        ("*", "Save current scene metaballs back to the .npz cache file"): "Сохранить текущие метаболы сцены в файл кэша .npz",

        # Dynamic runtime messages
        ("*", "Please select the source object!"): "Выберите исходный объект!",
        ("*", "Source object must be a mesh!"): "Исходный объект должен быть мешем!",
        ("*", "Please select the target object!"): "Выберите целевой объект!",
        ("*", "Target object must be a mesh!"): "Целевой объект должен быть мешем!",
        ("*", "Metaball collection MWC_Metaballs is empty or not found!"): "Коллекция метаболлов MWC_Metaballs пуста или не найдена!",
        ("*", "Successfully created {} metaballs."): "Успешно создано {} метаболлов.",
        ("*", "Weights successfully applied to target mesh."): "Веса успешно применены к целевому мешу.",
        ("*", "Cache successfully cleared."): "Кэш успешно очищен.",
        ("*", "Successfully spawned {} metaball objects in the viewport."): "Успешно отображено {} объектов метаболлов во вьюпорте.",
        ("*", "Viewport metaballs cleared."): "Метаболлы убраны из вьюпорта.",
    }
}

def t(msgid, *args, context='*'):
    """
    Translates string using Blender's native pgettext translation engine.
    Falls back to the original English string if translation is not registered or not active.
    Supports .format() style string formatting.
    """
    if not msgid:
        return ""
    try:
        translated = bpy.app.translations.pgettext(msgid, context)
    except Exception:
        translated = msgid
    
    if args:
        try:
            return translated.format(*args)
        except Exception:
            return translated
    return translated
