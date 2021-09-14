#!/usr/bin/env gxi
(import :std/text/json :std/text/utf8 :std/text/zlib)

(def (get-json-gz fn)
  (string->json-object
    (utf8->string
      (call-with-input-file fn uncompress))))
(def (hash-view h)
  (hash-map (lambda (k v)
              (cons k
                (cond
                  ((hash? v) (hash->list v))
                  ((list? v) ['list-with-length (length v)]) (else v))))
    h))
(def (prof-summary mem-prof)
  (hash-map
    (lambda (k v)
      [k
       (map (lambda (x)
              (if (hash? (cdr x))
                (set-cdr! x (hash->list (cdr x)))
                x))
         (hash->list (hash-ref v 'profileSummary)))])
    mem-prof))
(def (mem-per-allocator tbl)
  (hash-ref tbl 'memoryProfilePerAllocator))
(def (read-mem-prof tbl)
  (pp (hash-view tbl))
  (let (mem-prof (mem-per-allocator tbl))
    (pp (prof-summary mem-prof))
    (pp (map hash-view
          (take
            (hash-ref (hash-ref mem-prof 'GPU_0_bfc) 'memoryProfileSnapshots)
            7)))))

(def (main . args)
  (for-each
    (lambda (fn)
      (read-mem-prof (get-json-gz fn)))
    args))
