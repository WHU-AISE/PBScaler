

package org.myproject.ms.monitoring.instrument.msg;


public class TMHead {

	public static final String SPAN_ID_NAME = "spanId";
	public static final String SAMPLED_NAME = "spanSampled";
	public static final String PROCESS_ID_NAME = "spanProcessId";
	public static final String PARENT_ID_NAME = "spanParentSpanId";
	public static final String TRACE_ID_NAME = "spanTraceId";
	public static final String SPAN_NAME_NAME = "spanName";
	public static final String SPAN_FLAGS_NAME = "spanFlags";

	static final String MESSAGE_SENT_FROM_CLIENT = "messageSent";
	static final String HEADER_DELIMITER = "_";

	private TMHead() {}
}
