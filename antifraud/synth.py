"""Генератор синтетических CDR со схемами S1–S7 и ловушками D1–D5.

Параметры схем — из описания кейса (презентация Nexign). Параметр `scale`
сдвигает интенсивность схем (0.5 — мошенники вдвое «тише»), `dur_shift` —
длительности, `error_rate` — долю битых строк. Возвращает строки CDR и
истинную разметку {номер: (схема, роль)} — для честной оценки детекторов.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

CODES = {"CITY": ["842", "499", "892"], "ALFA": ["025", "026", "027"],
         "BARB": ["899", "810", "834"], "BELL": ["210", "214", "200"]}
MOBILE = CODES["ALFA"] + CODES["BARB"] + CODES["BELL"]
START = datetime(2026, 10, 1)
DAYS = 5


@dataclass
class Synth:
    seed: int = 7
    n_users: int = 6000
    n_city: int = 1200
    scale: float = 1.0
    dur_shift: float = 1.0
    error_rate: float = 0.0003
    rows: list = field(default_factory=list)
    truth: dict = field(default_factory=dict)      # num -> (scheme, role)
    legit: dict = field(default_factory=dict)      # num -> D#

    def __post_init__(self):
        self.r = random.Random(self.seed)
        self.used = set()
        self.imsi_of = {}

    # ---------- утилиты ----------
    def num(self, kind="mobile"):
        while True:
            code = self.r.choice(CODES["CITY"] if kind == "city" else MOBILE)
            n = code + f"{self.r.randrange(10 ** 7):07d}"
            if n not in self.used:
                self.used.add(n); return n

    def imsi(self, n):
        if n[:3] in CODES["CITY"]:
            return ""
        if n not in self.imsi_of:
            self.imsi_of[n] = f"2500{self.r.randrange(1, 4)}{self.r.randrange(10 ** 10):010d}"
        return self.imsi_of[n]

    def t(self, day=None, hour=None):
        d = self.r.randrange(DAYS) if day is None else day
        h = self.r.randrange(24) if hour is None else hour
        return START + timedelta(days=d, hours=h, seconds=self.r.randrange(3600))

    def call(self, a, b, ts, dur, typ=None, c="", imsi=None):
        if typ is None:
            typ = "01" if a[:3] in CODES["CITY"] else "02"
        im = self.imsi(a) if imsi is None else imsi
        self.rows.append((typ, im if typ != "01" else "", a, b, ts, max(0, int(dur)), c))

    def dur(self, lo, hi):
        return self.r.uniform(lo, hi) * self.dur_shift

    def k(self, lo, hi):
        return max(1, int(self.r.randint(lo, hi) * self.scale))

    # ---------- фон ----------
    def background(self):
        users = [self.num() for _ in range(self.n_users)]
        city = [self.num("city") for _ in range(self.n_city)]
        self.population = users + city
        for u in users + city:
            friends = self.r.sample(self.population, self.r.randint(3, 12))
            for _ in range(self.r.randint(3, 25)):
                b = self.r.choice(friends) if self.r.random() < 0.7 else self.r.choice(self.population)
                if b == u:
                    continue
                hour = self.r.choices(range(24), weights=[1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 7, 5, 3, 2])[0]
                d = 0 if self.r.random() < 0.08 else self.r.expovariate(1 / 90)
                self.call(u, b, self.t(hour=hour), d)
        # активные «корпоративные» абоненты: много звонков, есть входящие
        for _ in range(15):
            u = self.num()
            for _ in range(self.r.randint(150, 600)):
                b = self.r.choice(self.population)
                ts = self.t(hour=self.r.choices(range(24), weights=[1] * 7 + [6] * 12 + [2] * 5)[0])
                self.call(u, b, ts, self.r.expovariate(1 / 80))
                if self.r.random() < 0.25:
                    self.call(self.r.choice(self.population), u, ts + timedelta(minutes=7), self.r.expovariate(1 / 80))

    # ---------- схемы ----------
    def s1(self, n=6):
        for _ in range(n):
            src, prem = self.num(), self.num("city")
            self.truth[src] = ("S1", "wangiri_source"); self.truth[prem] = ("S1", "premium_C")
            t0 = self.t(hour=self.r.randrange(8, 21)); span = self.r.randint(30, 180)
            victims = self.r.sample(self.population, self.k(300, 800))
            for v in victims:
                ts = t0 + timedelta(seconds=self.r.randrange(span * 60))
                self.call(src, v, ts, self.r.randint(0, 2))
                if self.r.random() < 0.05:
                    self.call(v, src, ts + timedelta(minutes=self.r.randint(2, 120)), self.dur(60, 300), typ="03", c=prem)

    def s2(self, n=5):
        for _ in range(n):
            robots = [self.num() for _ in range(self.r.randint(1, 3))]
            for x in robots:
                self.truth[x] = ("S2", "robot_dialer")
            base = int(self.r.choice(MOBILE) + f"{self.r.randrange(10 ** 6, 9 * 10 ** 6):07d}")
            t0 = self.t(hour=self.r.randrange(9, 20)); span = self.r.randint(30, 120)
            total = self.k(700, 2500)
            for i in range(total):
                x = robots[i % len(robots)]
                ts = t0 + timedelta(seconds=int(span * 60 * i / total))
                self.call(x, f"{base + i:010d}", ts, 0 if self.r.random() < 0.25 else self.dur(4, 25))

    def s3(self, farms=4):
        for _ in range(farms):
            t0 = self.t()
            for _ in range(self.r.randint(3, 8)):
                sim = self.num(); self.truth[sim] = ("S3", "simbox_sim")
                ts = t0 + timedelta(minutes=self.r.randrange(120))
                for _ in range(self.k(100, 200)):
                    d = self.dur(30, 150)
                    self.call(sim, self.num(), ts, d)
                    ts += timedelta(seconds=d + self.r.randint(300, 700))

    def s4(self, n=4):
        for _ in range(n):
            x = self.num(); self.truth[x] = ("S4", "cloned_msisdn")
            imsis = [f"2500{self.r.randrange(1, 4)}{self.r.randrange(10 ** 10):010d}" for _ in range(self.r.randint(4, 10))]
            t0 = self.t()
            for _ in range(self.k(40, 120)):
                ts = t0 + timedelta(seconds=self.r.randrange(20 * 3600))
                self.call(x, self.r.choice(self.population), ts, self.dur(20, 200), imsi=self.r.choice(imsis))

    def s5(self, templates=4):
        for _ in range(templates):
            nodes = [self.num() if i % 2 else self.num("city") for i in range(self.r.randint(4, 7))]
            for nd in nodes:
                self.truth[nd] = ("S5", "forward_chain_node")
            for _ in range(self.k(30, 90)):
                a = self.r.choice(self.population); ts = self.t()
                length = self.r.randint(3, len(nodes))
                ring = self.r.random() < 0.3
                d = 0 if ring else self.dur(20, 200)
                for i in range(length - 1):
                    self.call(a, nodes[i], ts + timedelta(seconds=i), d, typ="03", c=nodes[i + 1])
                if ring:
                    self.call(a, nodes[length - 1], ts + timedelta(seconds=length), 0, typ="03", c=nodes[0])

    def s6(self, groups=4):
        for _ in range(groups):
            g = [self.num() for _ in range(self.r.randint(5, 10))]
            for x in g:
                self.truth[x] = ("S6", "carousel_member")
            day0 = self.r.randrange(2)
            for _ in range(self.k(400, 1000)):
                a, b = self.r.sample(g, 2)
                self.call(a, b, self.t(day=self.r.randrange(day0, DAYS)), self.dur(8, 45))

    def s7(self, n=5):
        for _ in range(n):
            fr = self.num(); spoofs = [self.num("city") for _ in range(self.r.randint(3, 6))]
            self.truth[fr] = ("S7", "vishing_caller")
            for s in spoofs:
                self.truth[s] = ("S7", "spoofed_city")
            for _ in range(self.k(15, 40)):
                v = self.r.choice(self.population); t_long = self.t(hour=self.r.randrange(9, 23))
                for _ in range(self.r.randint(4, 8)):
                    self.call(self.r.choice(spoofs), v, t_long - timedelta(minutes=self.r.randint(5, 180)), self.r.choice([0, 0, 3, 5, 8]))
                self.call(fr, v, t_long, self.dur(420, 1500))
                if self.r.random() < 0.5:
                    self.call(v, self.r.choice(spoofs), t_long + timedelta(minutes=self.r.randint(5, 90)), self.dur(10, 60))

    # ---------- легальные ловушки ----------
    def d1(self, n=3):
        for _ in range(n):
            x = self.num("city"); self.legit[x] = "D1"
            for day in (0, 1, 4):
                for _ in range(self.r.randint(300, 600)):
                    ts = self.t(day=day, hour=self.r.randrange(9, 20))
                    self.call(x, self.r.choice(self.population), ts, 0 if self.r.random() < 0.35 else self.dur(20, 300))
                    if self.r.random() < 0.25:
                        self.call(self.r.choice(self.population), x, ts + timedelta(minutes=3), self.dur(30, 300))

    def d2(self, n=2):
        for _ in range(n):
            x = self.num("city"); self.legit[x] = "D2"
            base = self.r.sample(self.population, 400); hours = self.r.sample(range(9, 21), 3)
            for day in range(DAYS):
                for h in hours:
                    for v in self.r.sample(base, self.r.randint(60, 100)):
                        self.call(x, v, START + timedelta(days=day, hours=h, seconds=self.r.randrange(1800)), self.dur(6, 20))

    def d3(self, n=2):
        for _ in range(n):
            g = [self.num() for _ in range(self.r.randint(15, 25))]
            for x in g:
                self.legit[x] = "D3"
            for _ in range(self.r.randint(400, 700)):
                a, b = self.r.sample(g, 2) if self.r.random() < 0.6 else (self.r.choice(g), self.r.choice(self.population))
                hour = self.r.randrange(0, 6) if self.r.random() < 0.6 else self.r.randrange(24)
                ts = self.t(hour=hour); self.call(a, b, ts, self.dur(30, 200))
                if self.r.random() < 0.3:
                    self.call(b, a, ts + timedelta(minutes=5), self.dur(30, 200))

    def d4(self, n=2):
        for _ in range(n):
            hot = self.num("city"); comp = [self.num("city") for _ in range(self.r.randint(10, 30))]
            self.legit[hot] = "D4"
            for c in comp:
                self.legit[c] = "D4"
            for _ in range(self.r.randint(300, 800)):
                self.call(self.r.choice(self.population), self.r.choice(comp), self.t(hour=self.r.randrange(8, 22)),
                          self.dur(30, 300), typ="03", c=hot)

    def d5(self, n=2):
        for _ in range(n):
            x = self.num("city"); self.legit[x] = "D5"
            t0 = self.t(hour=self.r.randrange(9, 19))
            for v in self.r.sample(self.population, self.r.randint(500, 800)):
                self.call(x, v, t0 + timedelta(seconds=self.r.randrange(3600)), self.dur(10, 40))

    # ---------- сборка ----------
    def build(self):
        self.background()
        for g in (self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7,
                  self.d1, self.d2, self.d3, self.d4, self.d5):
            g()
        self.rows.sort(key=lambda x: x[4])
        return self

    def lines(self):
        out = []
        for typ, im, a, b, ts, d, c in self.rows:
            line = f"{typ};{im};{a};{b};{ts:%Y%m%d%H%M%S};{d};{c}"
            if self.r.random() < self.error_rate:            # ошибки формата, как в реальных данных
                kind = self.r.randrange(5)
                line = [f"{typ};{im};+7{a};{b};{ts:%Y%m%d%H%M%S};{d};{c}",
                        f"{typ};{im};{a};{b};{ts:%Y-%m-%d %H:%M:%S};{d};{c}",
                        f"{typ};{im};{a};{b};{ts:%Y%m%d%H%M%S};abc;{c}",
                        f"{typ};{im};{a};{b};{ts:%Y%m%d%H%M%S};{d};{c};;",
                        f"AA;{im};{a};{b};{ts:%Y%m%d%H%M%S};{d};{c}"][kind]
            out.append(line + "\n")
        return out

    def sources(self, per_file=20000):
        import io
        ls = self.lines()
        return [(f"SYN_{i // per_file:04d}.cdr", io.StringIO("".join(ls[i:i + per_file]))) for i in range(0, len(ls), per_file)]
