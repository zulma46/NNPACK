#pragma once

#ifdef _WIN32
static inline void scalar_swap(float* a, float* b) {
#else
static inline void scalar_swap(float a[restrict static 1], float b[restrict static 1]) {
#endif
    const float new_a = *b;
    const float new_b = *a;
    *a = new_a;
    *b = new_b;
}

#ifdef _WIN32
static inline void scalar_butterfly(float* a, float* b) {
#else
static inline void scalar_butterfly(float a[restrict static 1], float b[restrict static 1]) {
#endif
    const float new_a = *a + *b;
    const float new_b = *a - *b;
    *a = new_a;
    *b = new_b;
}

#ifdef _WIN32
static inline void scalar_butterfly_and_negate_b(float* a, float* b) {
#else
static inline void scalar_butterfly_and_negate_b(float a[restrict static 1], float b[restrict static 1]) {
#endif
    const float new_a = *a + *b;
    const float new_b = *b - *a;
    *a = new_a;
    *b = new_b;
}

#ifdef _WIN32
static inline void scalar_butterfly_with_negated_b(float* a, float* b) {
#else
static inline void scalar_butterfly_with_negated_b(float a[restrict static 1], float b[restrict static 1]) {
#endif
    const float new_a = *a - *b;
    const float new_b = *a + *b;
    *a = new_a;
    *b = new_b;
}
