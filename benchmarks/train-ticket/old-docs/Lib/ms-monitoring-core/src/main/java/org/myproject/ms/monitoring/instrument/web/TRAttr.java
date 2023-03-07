

package org.myproject.ms.monitoring.instrument.web;


public final class TRAttr {

	
	public static final String HANDLED_SPAN_REQUEST_ATTR = TRAttr.class.getName()
			+ ".TRACE_HANDLED";

	
	public static final String ERROR_HANDLED_SPAN_REQUEST_ATTR = TRAttr.class.getName()
			+ ".ERROR_TRACE_HANDLED";

	
	public static final String NEW_SPAN_REQUEST_ATTR = TRAttr.class.getName()
			+ ".TRACE_HANDLED_NEW_SPAN";

	
	public static final String SPAN_CONTINUED_REQUEST_ATTR = TRAttr.class.getName()
					+ ".TRACE_CONTINUED";

	private TRAttr() {}
}
