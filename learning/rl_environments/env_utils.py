def reward_value(value, max_value=1, min_value=0, power=1, factor=1, invert=False):
	normalized_value = pow(abs(value), power) / pow(max_value, power)
	if (invert):
		normalized_value = 1 - normalized_value
	return max(normalized_value * factor, min_value)
