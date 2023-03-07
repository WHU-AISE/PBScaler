

package org.myproject.ms.monitoring.util;

import org.apache.commons.logging.Log;


public final class ExceptionUtils {
	private static final Log log = org.apache.commons.logging.LogFactory
			.getLog(ExceptionUtils.class);
	private static boolean fail = false;
	private static Exception lastException = null;

	private ExceptionUtils() {
		throw new IllegalStateException("Utility class can't be instantiated");
	}

	public static void warn(String msg) {
		log.warn(msg);
		if (fail) {
			IllegalStateException exception = new IllegalStateException(msg);
			ExceptionUtils.lastException = exception;
			throw exception;
		}
	}

	public static Exception getLastException() {
		return ExceptionUtils.lastException;
	}

	public static void setFail(boolean fail) {
		ExceptionUtils.fail = fail;
		ExceptionUtils.lastException = null;
	}

	public static String getExceptionMessage(Throwable e) {
		return e.getMessage() != null ? e.getMessage() : e.toString();
	}
}
