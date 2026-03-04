def compute_minimum_possible_cost(vendors, demand):
    remaining = demand
    total_cost = 0.0

    # Urutkan vendor dari termurah
    sorted_vendors = sorted(vendors, key=lambda v: v.cost_per_unit)

    for v in sorted_vendors:
        alloc = min(v.capacity, remaining)
        total_cost += alloc * v.cost_per_unit
        remaining -= alloc

        if remaining <= 0:
            break

    if remaining > 0:
        # Bahkan capacity total tidak cukup
        return float("inf")

    return total_cost
if __name__ == "__main__":
    print("Utils loaded successfully")