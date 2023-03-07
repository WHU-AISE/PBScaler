

package org.myproject.ms.monitoring.instrument.web;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;


class ServletUtils {

	static String getHeader(HttpServletRequest request, HttpServletResponse response,
			String name) {
		String value = request.getHeader(name);
		return value != null ? value : response.getHeader(name);
	}

}
