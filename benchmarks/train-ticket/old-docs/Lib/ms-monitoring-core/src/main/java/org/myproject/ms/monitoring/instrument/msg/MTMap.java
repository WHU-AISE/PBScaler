

package org.myproject.ms.monitoring.instrument.msg;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.myproject.ms.monitoring.ItemTextMap;
import org.springframework.messaging.Message;
import org.springframework.messaging.support.MessageBuilder;
import org.springframework.messaging.support.MessageHeaderAccessor;
import org.springframework.messaging.support.NativeMessageHeaderAccessor;
import org.springframework.util.StringUtils;


class MTMap implements ItemTextMap {

	private final MessageBuilder delegate;

	public MTMap(MessageBuilder delegate) {
		this.delegate = delegate;
	}

	@Override
	public Iterator<Map.Entry<String, String>> iterator() {
		Map<String, String> map = new HashMap<>();
		for (Map.Entry<String, Object> entry : this.delegate.build().getHeaders()
				.entrySet()) {
			map.put(entry.getKey(), String.valueOf(entry.getValue()));
		}
		return map.entrySet().iterator();
	}

	@Override
	@SuppressWarnings("unchecked")
	public void put(String key, String value) {
		if (!StringUtils.hasText(value)) {
			return;
		}
		Message<?> initialMessage = this.delegate.build();
		MessageHeaderAccessor accessor = MessageHeaderAccessor
				.getMutableAccessor(initialMessage);
		accessor.setHeader(key, value);
		if (accessor instanceof NativeMessageHeaderAccessor) {
			NativeMessageHeaderAccessor nativeAccessor = (NativeMessageHeaderAccessor) accessor;
			nativeAccessor.setNativeHeader(key, value);
		}
		this.delegate.copyHeaders(accessor.toMessageHeaders());
	}
}
