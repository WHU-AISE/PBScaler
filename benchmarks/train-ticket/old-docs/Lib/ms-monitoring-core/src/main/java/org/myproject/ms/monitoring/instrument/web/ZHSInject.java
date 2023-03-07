package org.myproject.ms.monitoring.instrument.web;

import java.util.Map;

import org.myproject.ms.monitoring.Item;
import org.myproject.ms.monitoring.ItemTextMap;
import org.springframework.util.StringUtils;


public class ZHSInject implements HSInject {

	private static final String HEADER_DELIMITER = "-";

	@Override
	public void inject(Item span, ItemTextMap carrier) {
		setHeader(carrier, Item.TRACE_ID_NAME, span.traceIdString());
		setIdHeader(carrier, Item.SPAN_ID_NAME, span.getSpanId());
		setHeader(carrier, Item.SAMPLED_NAME, span.isExportable() ? Item.SPAN_SAMPLED : Item.SPAN_NOT_SAMPLED);
		setHeader(carrier, Item.SPAN_NAME_NAME, span.getName());
		setIdHeader(carrier, Item.PARENT_ID_NAME, getParentId(span));
		setHeader(carrier, Item.PROCESS_ID_NAME, span.getProcessId());
		for (Map.Entry<String, String> entry : span.baggageItems()) {
			carrier.put(prefixedKey(entry.getKey()), entry.getValue());
		}
	}

	private String prefixedKey(String key) {
		if (key.startsWith(Item.SPAN_BAGGAGE_HEADER_PREFIX + HEADER_DELIMITER)) {
			return key;
		}
		return Item.SPAN_BAGGAGE_HEADER_PREFIX + HEADER_DELIMITER + key;
	}

	private Long getParentId(Item span) {
		return !span.getParents().isEmpty() ? span.getParents().get(0) : null;
	}

	private void setIdHeader(ItemTextMap carrier, String name, Long value) {
		if (value != null) {
			setHeader(carrier, name, Item.idToHex(value));
		}
	}

	private void setHeader(ItemTextMap carrier, String name, String value) {
		if (StringUtils.hasText(value)) {
			carrier.put(name, value);
		}
	}

}
