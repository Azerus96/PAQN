#pragma once

namespace ofc {
    // Единый источник правды для размера инфосета
    constexpr int INFOSET_SIZE = 1486;
    // НОВАЯ КОНСТАНТА: Размер вектора, описывающего действие
    // 52 карты * 4 возможных слота (top, middle, bottom, discard)
    constexpr int ACTION_VECTOR_SIZE = 208;
    // ДОБАВЛЕНА КОНСТАНТА для исправления ошибки компиляции
    constexpr int SUIT_COUNT = 4;
}
