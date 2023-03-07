

package org.myproject.ms.monitoring.instrument.web;

import javax.servlet.http.HttpServletRequest;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.myproject.ms.monitoring.ItemTextMap;
import org.springframework.web.util.UrlPathHelper;


class HSRTMap implements ItemTextMap {

	private final HttpServletRequest delegate;
	private final Map<String, String> additionalHeaders = new HashMap<>();

	HSRTMap(HttpServletRequest delegate) {
		this.delegate = delegate;
		UrlPathHelper urlPathHelper = new UrlPathHelper();
		this.additionalHeaders.put(ZHSExtra.URI_HEADER,
				urlPathHelper.getPathWithinApplication(delegate));
	}

	@Override
	public Iterator<Map.Entry<String, String>> iterator() {
		Map<String, String> map = new HashMap<>();
		Enumeration<String> headerNames = this.delegate.getHeaderNames();
		while (headerNames != null && headerNames.hasMoreElements()) {
			String name = headerNames.nextElement();
			map.put(name, this.delegate.getHeader(name));
		}
		map.putAll(this.additionalHeaders);
		return map.entrySet().iterator();
	}

	@Override
	public void put(String key, String value) {
		this.additionalHeaders.put(key, value);
	}
}
