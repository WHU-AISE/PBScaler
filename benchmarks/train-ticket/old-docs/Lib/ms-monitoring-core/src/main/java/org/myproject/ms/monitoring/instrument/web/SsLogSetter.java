

package org.myproject.ms.monitoring.instrument.web;

import java.lang.invoke.MethodHandles;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.myproject.ms.monitoring.Item;


class SsLogSetter {

	private static final Log log = LogFactory.getLog(MethodHandles.lookup().lookupClass());

	static void annotateWithServerSendIfLogIsNotAlreadyPresent(Item span) {
		if (span == null) {
			return;
		}
		for (org.myproject.ms.monitoring.Log log1 : span.logs()) {
			if (Item.SERVER_SEND.equals(log1.getEvent())) {
				if (log.isTraceEnabled()) {
					log.trace("Span was already annotated with SS, will not do it again");
				}
				return;
			}
		}
		if (log.isTraceEnabled()) {
			log.trace("Will set SS on the span");
		}
		span.logEvent(Item.SERVER_SEND);
	}
}
