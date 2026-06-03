import statistics


class BeatStatistics:

    @staticmethod
    def build(beats):

        rr = [
            b["rr_interval"]
            for b in beats
            if b["rr_interval"] is not None
        ]

        hr = [
            b["heart_rate"]
            for b in beats
            if b["heart_rate"] is not None
        ]

        return {

            "rr_mean":
                statistics.mean(rr),

            "rr_std":
                statistics.stdev(rr)
                if len(rr) > 1 else 0,

            "rr_min":
                min(rr),

            "rr_max":
                max(rr),

            "hr_mean":
                statistics.mean(hr),

            "hr_min":
                min(hr),

            "hr_max":
                max(hr)
        }